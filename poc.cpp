/////////////////////////
// Copyright: CERN
// Author: Serena Ziviani
/////////////////////////

#include <iostream>
#include <mpi.h>
#include <tbb/tbb.h>
#include <unistd.h>
#include <vector>
#include <stack>

const int DIETAG = 2048;
const int WORKTAG = 1024;

int MPI_MASTER_THREAD;

void check_error(int ret, std::string err_string)
{
    if (ret != MPI_SUCCESS) {
        std::cout << err_string << ": " << ret << std::endl;
        exit(1);
    }
}

class Scheduler
{
    private:
        int nprocs_;
        std::stack<int> gpu_machines_;
        tbb::mutex m_;
    public:
        Scheduler(int nprocs) {
            nprocs_ = nprocs;
            gpu_machines_.push(1);
            gpu_machines_.push(0);
        }
        int get_id() {
            m_.lock();
            int tmp = gpu_machines_.top();
            gpu_machines_.pop();
            m_.unlock();
            return tmp;
        }
};

Scheduler *scheduler;

/* Create and send a task to the GPU */
struct Task {
    private:
        int id_;
        bool offloadable_;
        std::string name_;

        void function () {
            std::cout << "CPU task executing" << std::endl;
        }
    public:
        Task(int id, bool offloadable, std::string name) {
            id_ = id;
            offloadable_ = offloadable;
            name_ = name;
        }
        bool operator()(tbb::mutex &m) {
            if (offloadable_) {
                m.lock();
                int gpu_pe = scheduler->get_id();
                m.unlock();

                std::cout << std::string(2*(id_ + 1), '\t') << "MST " << id_ << ": Sending " << name_ << " to node " << gpu_pe << std::endl;
                int ret = MPI_Send((void *)name_.c_str(), name_.size() + 1, MPI_CHAR, gpu_pe, WORKTAG + id_, MPI_COMM_WORLD);
                std::cout << std::string(2*(id_ + 1), '\t') << "MST " << id_ << ": Sent " << name_ << " to node " << gpu_pe << " tag " << WORKTAG + id_ << std::endl;
                check_error(ret, "Error in MPI_Send, master thread");

                MPI_Status status;
                ret = MPI_Recv(0, 0, MPI_INT, gpu_pe, WORKTAG + id_, MPI_COMM_WORLD, &status);
                std::cout << std::string(2*(id_ + 1), '\t') << "MST " << id_ << ": Received answer from " << gpu_pe << " tag " << WORKTAG + id_ << std::endl;
                check_error(ret, "Error in MPI_Recv, master thread");

                /* do other work */
                return true;
            } else {
                function();
                return false;
            }
        }
};

struct Receiver {
    private:
        int id_;
        int pool_size_;
        int mpi_id_;
    public:
        Receiver(int id, int pool_size, int mpi_id) {
            id_ = id;
            pool_size_ = pool_size;
            mpi_id_ = mpi_id;
        }
        void operator()() {

            int count, ret;
            MPI_Status status;
            MPI_Request request;
            int abs_id = mpi_id_ * pool_size_ + id_;

            while (id_ == 0) {
                std::cout << std::string(mpi_id_, '\t') << "SLAVE " << mpi_id_ << " probing for work " << std::endl;
                ret = MPI_Probe(MPI_MASTER_THREAD, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
                check_error(ret, "Error in MPI_Probe, abs_id " + abs_id);
                std::cout << std::string(mpi_id_, '\t') << "SLAVE " << mpi_id_ << " probed for tag " << status.MPI_TAG << ", node " << status.MPI_SOURCE << std::endl;

                if (status.MPI_TAG == DIETAG) {
                    std::cout << std::string(mpi_id_, '\t') << "SLAVE " << mpi_id_ << " exiting " << std::endl;
                    return;
                }

                ret = MPI_Get_count(&status, MPI_CHAR, &count);
                check_error(ret, "Error in MPI_Get_count, abs_id" + abs_id);

                char *rec_buf = new char[count];
                ret = MPI_Recv(static_cast<void *>(rec_buf), count, MPI_CHAR, status.MPI_SOURCE, status.MPI_TAG, MPI_COMM_WORLD, &status);
                check_error(ret, "Error in MPI_Recv, abs_id " + abs_id);

                std::cout << std::string(mpi_id_, '\t') << "SLAVE " << mpi_id_ << " received work " << rec_buf << " from " << status.MPI_SOURCE << std::endl;
                MPI_Send(0, 0, MPI_INT, status.MPI_SOURCE, status.MPI_TAG, MPI_COMM_WORLD);
                std::cout << std::string(mpi_id_, '\t') << "SLAVE " << mpi_id_ << " sent answer to node " << status.MPI_SOURCE << " tag " << status.MPI_TAG << std::endl;
            }
      }
};

int main(int argc, char *argv[])
{
    // initialization
    tbb::task_scheduler_init init();

    int ret;
    int mpi_rank, mpi_id, hostname_len;
    char hostname[MPI_MAX_PROCESSOR_NAME];

    ret = MPI_Init(&argc, &argv);
    check_error(ret, "Error in MPI_Init");

    ret = MPI_Comm_size(MPI_COMM_WORLD, &mpi_rank);
    check_error(ret, "Error in MPI_Comm_size");
    ret = MPI_Comm_rank(MPI_COMM_WORLD, &mpi_id);
    check_error(ret, "Error in MPI_Comm_rank");
    MPI_Get_processor_name(hostname, &hostname_len);

    MPI_MASTER_THREAD = mpi_rank - 1;
    scheduler = new Scheduler(mpi_rank);

    if (mpi_id == MPI_MASTER_THREAD) { // Producer

        tbb::mutex m;

        std::vector<Task> tasks;
        // FIXME needs a nr of GPU tasks equal to the nr of mpi * tbb tasks
        tasks.push_back(Task(0, true, "binary_search"));
        tasks.push_back(Task(1, true, "matr_mul"));
        //tasks.push_back(Task(2, true, "conv"));
        //tasks.push_back(Task(3, true, "riemann"));

        tbb::parallel_for_each(tasks.begin(), tasks.end(), [&](Task &t) { t(m); } );
        std::cout << "MST: Exited from tbb::parallel_for_each" << std::endl;

        // Now stop all the GPU machines

        for(int i = 0; i < mpi_rank - 1; i++) {
            std::cout << "MST: Send exit tag to " << i << std::endl;
            MPI_Send(0, 0, MPI_INT, i, DIETAG, MPI_COMM_WORLD);
        }

    } else { // GPU machines - consumers

        std::vector<Receiver> receivers;
        receivers.push_back(Receiver(0, 2, mpi_id));
        receivers.push_back(Receiver(1, 2, mpi_id));

        tbb::parallel_for_each(receivers.begin(), receivers.end(), [&](Receiver &r) { r(); } );
    }

    ret = MPI_Finalize();
    check_error(ret, "Error in MPI_Finalize");
}

