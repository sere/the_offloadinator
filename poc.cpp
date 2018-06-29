/////////////////////////
// Copyright: CERN
// Author: Serena Ziviani
/////////////////////////

#include <iostream>
#include <mpi.h>
#include <tbb/tbb.h>
#include <unistd.h>
#include <vector>

void check_error(int ret, std::string err_string)
{
    if (ret != MPI_SUCCESS) {
        std::cout << err_string << ": " << ret << std::endl;
    }
}

struct Task {
    private:
        bool offloadable_;
        std::string name_;
        void function () {
            std::cout << "CPU task executing" << std::endl;
        }
    public:
        Task(bool offloadable, std::string name) {
            offloadable_ = offloadable;
            name_ = name;
        }
        bool operator()(int i) {
            if (offloadable_) {
                std::cout << "Sending computation to task " << i << " on a GPU-provided machine" << std::endl;
                int ret = MPI_Send((void *)name_.c_str(), name_.size() + 1, MPI_CHAR, i, i, MPI_COMM_WORLD);
                check_error(ret, "Error in MPI_Send, mpi_id 0");
                return true;
            } else {
                function();
                return false;
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

    if (mpi_id == 0) { // I am the one that offloads tasks

        std::vector<Task> tasks;
        tasks.push_back(Task(false, "binary_search"));
        tasks.push_back(Task(true, "matr_mul"));
        tasks.push_back(Task(true, "conv"));
        tasks.push_back(Task(true, "riemann"));

        tbb::parallel_for_each(tasks.begin(), tasks.end(), [&](Task &t) { static int i = 0; t(i); i++; } );
    } else { // GPU machines
        std::cout << "Machine " << mpi_id << " waiting for tasks..." << std::endl;

        int count;
        MPI_Status status;
        MPI_Request request;

        ret = MPI_Probe(0, mpi_id, MPI_COMM_WORLD, &status);
        check_error(ret, "Error in MPI_Probe, mpi_id " + mpi_id);
        std::cout << "mpi_id: " << mpi_id << std::endl;
        ret = MPI_Get_count(&status, MPI_CHAR, &count);
        check_error(ret, "Error in MPI_Get_count, mpi_id" + mpi_id);
        std::cout << "mpi_id: " << mpi_id << " count " << count << std::endl;

        char *rec_buf = new char[count];
        // Terrible, terrible hack in the next row
        ret = MPI_Irecv(static_cast<void *>(rec_buf), count, MPI_CHAR, 0, mpi_id, MPI_COMM_WORLD, &request);
        check_error(ret, "Error in MPI_Recv, mpi_id 0");
        std::cout << "Received work " << rec_buf << ", processing" << std::endl;
    }


    ret = MPI_Finalize();
    check_error(ret, "Error in MPI_Finalize");
}

