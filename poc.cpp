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
struct Producer {
    private:
        int id_;
        bool offloadable_;
        std::string name_;

        void function () {
            std::cout << "CPU task executing" << std::endl;
        }
    public:
        Producer(int id, bool offloadable, std::string name) {
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
                MPI_Send((void *)name_.c_str(), name_.size() + 1, MPI_CHAR, gpu_pe, WORKTAG + id_, MPI_COMM_WORLD);
                std::cout << std::string(2*(id_ + 1), '\t') << "MST " << id_ << ": Sent " << name_ << " to node " << gpu_pe << " tag " << WORKTAG + id_ << std::endl;

                MPI_Status status;
                MPI_Recv(0, 0, MPI_INT, gpu_pe, WORKTAG + id_, MPI_COMM_WORLD, &status);
                std::cout << std::string(2*(id_ + 1), '\t') << "MST " << id_ << ": Received answer from " << gpu_pe << " tag " << WORKTAG + id_ << std::endl;

                /* do other work */
                return true;
            } else {
                function();
                return false;
            }
        }
};

struct Consumer {
    private:
        int id_;
        int pool_size_;
        int mpi_id_;
    public:
        Consumer(int id, int pool_size, int mpi_id) {
            id_ = id;
            pool_size_ = pool_size;
            mpi_id_ = mpi_id;
        }
        void operator()() {

            int count;
            MPI_Status status;
            MPI_Message message;

            while (id_ == 0) {
                std::cout << std::string(mpi_id_, '\t') << "SLAVE " << mpi_id_ << " probing for work " << std::endl;
                MPI_Mprobe(MPI_MASTER_THREAD, MPI_ANY_TAG, MPI_COMM_WORLD, &message, &status);
                std::cout << std::string(mpi_id_, '\t') << "SLAVE " << mpi_id_ << " probed tag " << status.MPI_TAG << ", node " << status.MPI_SOURCE << std::endl;

                if (status.MPI_TAG == DIETAG) {
                    std::cout << std::string(mpi_id_, '\t') << "SLAVE " << mpi_id_ << " exiting " << std::endl;
                    return;
                }

                MPI_Get_count(&status, MPI_CHAR, &count);

                char *rec_buf = new char[count];
                MPI_Mrecv(static_cast<void *>(rec_buf), count, MPI_CHAR, &message, &status);
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

    int provided;
    int mpi_rank, mpi_id, hostname_len;
    char hostname[MPI_MAX_PROCESSOR_NAME];

    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

    MPI_Comm_size(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_id);
    MPI_Get_processor_name(hostname, &hostname_len);

    MPI_MASTER_THREAD = mpi_rank - 1;
    scheduler = new Scheduler(mpi_rank);

    if (mpi_id == MPI_MASTER_THREAD) { // Producer

        tbb::mutex m;

        std::vector<Producer> producers;
        producers.push_back(Producer(0, true, "binary_search"));
        producers.push_back(Producer(1, true, "matr_mul"));
        //producers.push_back(Producer(2, true, "conv"));
        //producers.push_back(Producer(3, true, "riemann"));

        tbb::parallel_for_each(producers.begin(), producers.end(), [&](Producer &p) { p(m); } );
        std::cout << "MST: Exited from tbb::parallel_for_each" << std::endl;

        // Now stop all the GPU nodes

        for(int i = 0; i < mpi_rank - 1; i++) {
            std::cout << "MST: Send exit tag to " << i << std::endl;
            MPI_Send(0, 0, MPI_INT, i, DIETAG, MPI_COMM_WORLD);
        }

    } else { // GPU nodes - consumers

        std::vector<Consumer> consumers;
        consumers.push_back(Consumer(0, 2, mpi_id));
        consumers.push_back(Consumer(1, 2, mpi_id));

        tbb::parallel_for_each(consumers.begin(), consumers.end(), [&](Consumer &c) { c(); } );
    }

    MPI_Finalize();
}

