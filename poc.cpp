/////////////////////////
// Copyright: CERN
// Author: Serena Ziviani
/////////////////////////

#include <iostream>
#include <mpi.h>
#include <tbb/tbb.h>
#include <unistd.h>
#include <vector>

int MPI_MASTER_THREAD;

void check_error(int ret, std::string err_string)
{
    if (ret != MPI_SUCCESS) {
        std::cout << err_string << ": " << ret << std::endl;
        exit(1);
    }
}

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
        bool operator()() {
            if (offloadable_) {
                std::cout << "Sending computation to task " << id_ << " on a GPU-provided machine" << std::endl;
                // FIXME id_/2 is a workaround; do instead: int gpu_tag = scheduler.get_tag() somewhere
                int ret = MPI_Send((void *)name_.c_str(), name_.size() + 1, MPI_CHAR, id_/2, id_, MPI_COMM_WORLD);
                check_error(ret, "Error in MPI_Send, mpi_id 0");
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

            std::cout << "Task " << id_ << " on machine " << mpi_id_ << ", absolute id " << abs_id << " waiting for tasks..." << std::endl;
            ret = MPI_Probe(MPI_MASTER_THREAD, abs_id , MPI_COMM_WORLD, &status);
            check_error(ret, "Error in MPI_Probe, abs_id " + abs_id);
            std::cout << "abs_id: " << abs_id << std::endl;
            ret = MPI_Get_count(&status, MPI_CHAR, &count);
            check_error(ret, "Error in MPI_Get_count, abs_id" + abs_id);

            char *rec_buf = new char[count];
            ret = MPI_Irecv(static_cast<void *>(rec_buf), count, MPI_CHAR, MPI_MASTER_THREAD, abs_id, MPI_COMM_WORLD, &request);
            check_error(ret, "Error in MPI_Recv, abs_id " + abs_id);
            std::cout << "Received work " << rec_buf << ", processing" << std::endl;
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

    if (mpi_id == MPI_MASTER_THREAD) { // I am the one that offloads tasks

        std::vector<Task> tasks;
        // FIXME needs a nr of GPU tasks equal to the nr of mpi * tbb tasks
        tasks.push_back(Task(0, true, "binary_search"));
        tasks.push_back(Task(1, true, "matr_mul"));
        tasks.push_back(Task(2, true, "conv"));
        tasks.push_back(Task(3, true, "riemann"));

        tbb::parallel_for_each(tasks.begin(), tasks.end(), [&](Task &t) { t(); } );

    } else { // GPU machines

        std::vector<Receiver> receivers;
        receivers.push_back(Receiver(0, 2, mpi_id));
        receivers.push_back(Receiver(1, 2, mpi_id));

        tbb::parallel_for_each(receivers.begin(), receivers.end(), [&](Receiver &r) { r(); } );
    }

    ret = MPI_Finalize();
    check_error(ret, "Error in MPI_Finalize");
}

