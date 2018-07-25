/////////////////////////
// Copyright: CERN
// Author: Serena Ziviani
/////////////////////////

#include <cassert>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <map>
#include <mpi.h>
#include <tbb/tbb.h>
#include <thread>
#include <unistd.h>
#include <vector>

const int WORKTAG = 1 << 10;
const int DIETAG = 1 << 11;
const int SCHED_REQ = 1 << 12;
const int SCHED_UPD = 1 << 13;
const int SCHED_DIE = DIETAG;

int MPI_MASTER_THREAD;
int MPI_SCHEDULER;
int MPI_FIRST_CPU;

class Scheduler {
  private:
    int n_gpu_nodes_;
    std::map<const int, int> gpu_nodes_;
    tbb::mutex m_;

    void print_state() {
        std::cout << "**** SCHED state ****" << std::endl;
        for (auto e : gpu_nodes_)
            std::cout << "* node " << e.first << " has " << e.second
                      << " GPUs *" << std::endl;
        std::cout << "*********************" << std::endl;
    }
    int use_node() {
        int k;
        auto it = gpu_nodes_.begin();
        if (it != gpu_nodes_.end()) {
            k = it->first;
            gpu_nodes_[k]--;
            if (gpu_nodes_[k] == 0) {
                std::cout << "SCHED erasing node " << k << std::endl;
                gpu_nodes_.erase(k);
            }
        } else {
            k = -1;
        }
        return k;
    }
    void update_node(int *nodeinfo) { gpu_nodes_[nodeinfo[0]] = nodeinfo[1]; }

  public:
    Scheduler(int n_gpu_nodes) {
        n_gpu_nodes_ = n_gpu_nodes;
        int nodeinfo[2];

        for (int i = 0; i <= n_gpu_nodes_ - 1; i++) {
            MPI_Recv((void *)nodeinfo, 2, MPI_INT, i, SCHED_UPD, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
            assert(i == nodeinfo[0]);
            gpu_nodes_[i] = nodeinfo[1];
        }
    }
    void manage_connections() {
        MPI_Message message;
        MPI_Status status;
        int node;
        int nodeinfo[2];

        while (true) {
            print_state();
            MPI_Mprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &message,
                       &status);
            std::cout << "SCHED probed message with tag " << status.MPI_TAG
                      << " from node " << status.MPI_SOURCE << std::endl;

            if (status.MPI_TAG / SCHED_REQ == 1) {
                MPI_Mrecv(NULL, 0, MPI_INT, &message, &status);
                node = use_node();
                std::cout << "SCHED send node " << node << " to node "
                          << status.MPI_SOURCE << "-"
                          << status.MPI_TAG % SCHED_REQ << std::endl;
                MPI_Send(&node, 1, MPI_INT, status.MPI_SOURCE, status.MPI_TAG,
                         MPI_COMM_WORLD);
            } else if (status.MPI_TAG / SCHED_UPD == 1) {
                MPI_Mrecv((void *)nodeinfo, 2, MPI_INT, &message, &status);
                std::cout << "SCHED update node " << nodeinfo[0] << "-"
                          << status.MPI_TAG % SCHED_REQ << " from "
                          << gpu_nodes_[nodeinfo[0]] << " to " << nodeinfo[1]
                          << std::endl;
                update_node(nodeinfo);
            } else if (status.MPI_TAG / SCHED_DIE == 1) {
                MPI_Mrecv(0, 0, MPI_INT, &message, &status);
                std::cout << "Scheduler exiting" << std::endl;
                return;
            } else {
                std::cout << "Wrong tag received by scheduler" << std::endl;
                return;
            }
        }
    }
};

Scheduler *scheduler;

/* Create and send a task to the GPU */
struct Producer {
  private:
    int id_;
    int mpi_id_;
    bool offloadable_;
    std::string name_;

    void function() {
        std::cout << "\t"
                  << "CPU " << mpi_id_ << "-" << id_
                  << ": task executing on CPU" << std::endl;
    }
    int get_node() {
        int node;
        MPI_Status status;
        MPI_Send(NULL, 0, MPI_INT, MPI_SCHEDULER, SCHED_REQ + id_,
                 MPI_COMM_WORLD);
        std::cout << "\t"
                  << "CPU " << mpi_id_ << "-" << id_
                  << ": Sent request for gpu node to scheduler " << std::endl;
        MPI_Recv(&node, 1, MPI_INT, MPI_SCHEDULER, SCHED_REQ + id_,
                 MPI_COMM_WORLD, &status);
        std::cout << "\t"
                  << "CPU " << mpi_id_ << "-" << id_ << ": Received node "
                  << node << " from scheduler " << std::endl;
        return node;
    }

  public:
    Producer(int id, int mpi_id, bool offloadable, std::string name) {
        id_ = id;
        mpi_id_ = mpi_id;
        offloadable_ = offloadable;
        name_ = name;
    }
    bool operator()() {
        if (offloadable_) {

            int gpu_pe = get_node();

            if (gpu_pe == -1) { // No GPU is available right now
                std::cout << "\t"
                          << "CPU " << mpi_id_ << "-" << id_
                          << ": No GPU available, executing task on CPU"
                          << std::endl;
                function();
                return false;
            }
            std::cout << "\t"
                      << "CPU " << mpi_id_ << "-" << id_ << ": Sending "
                      << name_ << " to node " << gpu_pe << std::endl;
            MPI_Send((void *)name_.c_str(), name_.size() + 1, MPI_CHAR, gpu_pe,
                     WORKTAG + id_ * 4 + mpi_id_, MPI_COMM_WORLD);
            std::cout << "\t"
                      << "CPU " << mpi_id_ << "-" << id_ << ": Sent " << name_
                      << " to node " << gpu_pe << " tag " << WORKTAG + id_
                      << std::endl;

            MPI_Status status;
            MPI_Recv(0, 0, MPI_INT, gpu_pe, WORKTAG + id_ * 4 + mpi_id_,
                     MPI_COMM_WORLD, &status);
            std::cout << "\t"
                      << "CPU " << mpi_id_ << "-" << id_
                      << ": Received answer from " << gpu_pe << " tag "
                      << WORKTAG + id_ << std::endl;

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
    int gpu_total_;
    int gpu_available_;

    void update_available_gpu(int gpus) {
        assert(gpus <= gpu_total_ && gpus >= 0);
        std::cout << "\t\t"
                  << "GPU " << mpi_id_ << ": update available gpus from "
                  << gpu_available_ << " to " << gpus << std::endl;
        gpu_available_ = gpus;
    }
    void send_gpu_info() {
        int nodeinfo[2] = {mpi_id_, gpu_available_};
        std::cout << "\t\t"
                  << "GPU " << mpi_id_ << " updating available gpu to "
                  << gpu_available_ << std::endl;
        MPI_Send((void *)nodeinfo, 2, MPI_INT, MPI_SCHEDULER, SCHED_UPD + id_,
                 MPI_COMM_WORLD);
    }

  public:
    Consumer(int id, int pool_size, int mpi_id) {
        id_ = id;
        pool_size_ = pool_size;
        mpi_id_ = mpi_id;
        gpu_total_ = std::rand() % 4 + 1;
        gpu_available_ = gpu_total_;
        send_gpu_info();
    }
    void operator()() {

        int count;
        MPI_Status status;
        MPI_Message message;

        while (id_ == 0) {
            std::cout << "\t\t"
                      << "GPU " << mpi_id_ << " probing for work" << std::endl;
            MPI_Mprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &message,
                       &status);
            std::cout << "\t\t"
                      << "GPU " << mpi_id_ << " probed tag " << status.MPI_TAG
                      << " from node " << status.MPI_SOURCE << std::endl;

            MPI_Get_count(&status, MPI_CHAR, &count);

            char *rec_buf = new char[count];
            MPI_Mrecv(static_cast<void *>(rec_buf), count, MPI_CHAR, &message,
                      &status);

            if (status.MPI_TAG == DIETAG) {
                std::cout << "\t\t"
                          << "GPU " << mpi_id_ << " exiting " << std::endl;
                return;
            }

            std::cout << "\t\t"
                      << "GPU " << mpi_id_ << " received work " << rec_buf
                      << " from node " << status.MPI_SOURCE << std::endl;
            update_available_gpu(gpu_available_ - 1);
            MPI_Send(0, 0, MPI_INT, status.MPI_SOURCE, status.MPI_TAG,
                     MPI_COMM_WORLD);
            std::this_thread::sleep_for(
                std::chrono::milliseconds((std::rand() % 100 + 1) * 10));
            std::cout << "\t\t"
                      << "GPU " << mpi_id_ << " sent answer to node "
                      << status.MPI_SOURCE << " tag " << status.MPI_TAG
                      << std::endl;
            update_available_gpu(gpu_available_ + 1);
            send_gpu_info();
        }
    }
};

int main(int argc, char *argv[]) {
    // initialization
    tbb::task_scheduler_init init();

    int provided;
    int mpi_rank, mpi_id, hostname_len;
    char hostname[MPI_MAX_PROCESSOR_NAME];
    int n;
    std::vector<int> ranks;

    std::srand(std::time(nullptr));

    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

    MPI_Comm_size(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_id);
    MPI_Get_processor_name(hostname, &hostname_len);

    if (mpi_rank < 3) {
        if (mpi_id == MPI_MASTER_THREAD) {
            std::cout << "Too few mpi nodes specified, exiting" << std::endl;
        }
        goto err;
    }

    MPI_SCHEDULER = mpi_rank - 1;
    MPI_MASTER_THREAD = mpi_rank - 2;
    MPI_FIRST_CPU = mpi_rank / 2; // TODO

    MPI_Group world_group;
    MPI_Comm_group(MPI_COMM_WORLD, &world_group);

    n = MPI_MASTER_THREAD - MPI_FIRST_CPU + 1;
    for (int i = 0; i < n; i++) {
        ranks.push_back(MPI_FIRST_CPU + i);
        std::cout << ranks[i] << ", ";
    }
    std::cout << std::endl;

    MPI_Group cpu_group;
    MPI_Group_incl(world_group, n, ranks.data(), &cpu_group);
    MPI_Comm CPU_comm;
    MPI_Comm_create_group(MPI_COMM_WORLD, cpu_group, 0, &CPU_comm);

    if (mpi_id == MPI_SCHEDULER) {

        std::cout << std::endl;
        std::cout << "##### cluster #####" << std::endl;
        std::cout << "# SCHED node " << MPI_SCHEDULER << "    #" << std::endl;
        std::cout << "# CPU nodes [" << MPI_MASTER_THREAD << ","
                  << MPI_FIRST_CPU << "] #" << std::endl;
        std::cout << "# GPU nodes [" << MPI_FIRST_CPU - 1 << ",0] #"
                  << std::endl;
        std::cout << "###################" << std::endl;
        std::cout << std::endl;

        scheduler = new Scheduler(MPI_FIRST_CPU);
        scheduler->manage_connections();

    } else if (mpi_id >= MPI_FIRST_CPU) { // Producer

        std::vector<Producer> producers;
        producers.push_back(Producer(0, mpi_id, true, "binary_search"));
        producers.push_back(Producer(1, mpi_id, true, "matr_mul"));
        producers.push_back(Producer(2, mpi_id, true, "conv"));
        producers.push_back(Producer(3, mpi_id, true, "riemann"));

        tbb::parallel_for_each(producers.begin(), producers.end(),
                               [&](Producer &p) { p(); });

        MPI_Barrier(CPU_comm);
        // Now stop all the GPU nodes
        if (mpi_id == MPI_MASTER_THREAD) {
            std::cout << "\t"
                      << "MST: Exited from tbb::parallel_for_each" << std::endl;
            for (int i = 0; i < MPI_FIRST_CPU; i++) {
                std::cout << "\t"
                          << "MST: Send exit tag to " << i << std::endl;
                MPI_Send(0, 0, MPI_INT, i, DIETAG, MPI_COMM_WORLD);
            }
            // Stop the scheduler
            // this instruction order is the only guarantee that the scheduler
            // is exiting after the computation (blocking send for stopping
            // gpus)
            MPI_Send(0, 0, MPI_INT, MPI_SCHEDULER, SCHED_DIE, MPI_COMM_WORLD);
        }

    } else { // GPU nodes - consumers

        std::vector<Consumer> consumers;
        consumers.push_back(Consumer(0, 2, mpi_id));
        consumers.push_back(Consumer(1, 2, mpi_id));

        tbb::parallel_for_each(consumers.begin(), consumers.end(),
                               [&](Consumer &c) { c(); });
    }

err:
    MPI_Finalize();
}
