using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using DequeNet;

namespace Namespace
{

    

    using System;

    using System.Collections.Generic;

    public static class Module
    {

        public class DQN
        {
            private int batch_size;
            private double gamma;
            private double exploration_max;
            private double exploration_min;
            private double exploration_decay;
            private double learning_rate;
            private double tau;
            private int n_actions;
            private int n_inputs;
            private List<object> loss_history;
            private int fit_count;
            private List<object> nodes_queue;
            private int memory_size;
            private Deque<(List<List<int>>, int, int, List<List<int>>, bool)> memory;


            public DQN(

                int memory_size,
                int batch_size,
                double gamma,
                double exploration_max,
                double exploration_min,
                double exploration_decay,
                double learning_rate,
                double tau,
                int n_actions,
                int n_inputs)
            {
                this.memory = new Deque<(List<List<int>>, int, int, List<List<int>>, bool)>();
                memory.Capacity = memory_size;
                this.batch_size = batch_size;
                this.gamma = gamma;
                this.exploration_max = exploration_max;
                this.exploration_min = exploration_min;
                this.exploration_decay = exploration_decay;
                this.learning_rate = learning_rate;
                this.tau = tau;
                this.loss_history = new List<object>();
                this.fit_count = 0;
                this.nodes_queue = new List<object>();
            }

            public virtual object step(List<List<int>> state, int action)
            {

                List<List<int>> next_state = new List<List<int>>(state);

                if (action == 0)
                {
                    next_state[0][0] -= 1;
                    // andei pra esquerda
                }
                else
                {
                    next_state[0][0] += 1;
                    // andei pra direita
                }
                var done = false;
                if (next_state[0][0] == 0 || next_state[0][0] == 3)
                {
                    done = true;

                }
                if (next_state[0][1] == 1 && next_state[0][0] == 1)
                {
                    next_state[0][1] = 0;
                    // se ta em cima do diamante e ele ainda existe, entao ele nao existe mais
                }
                return Tuple.Create(next_state, calc_reward(state, next_state, done), done);
            }

            public virtual object calc_reward(List<List<int>> state, List<List<int>> next_state, bool done)
            {
                if (next_state[0][0] == 0)
                {
                    return -10;
                }
                else if (next_state[0][0] == 3)
                {
                    return 10;
                }
                else if (state[0][1] == 1 && next_state[0][0] == 1)
                {
                    return 3;
                }
                else
                {
                    return 0;
                }
            }

            public virtual object reset_env()
            {
                var done = false;
                List<List<int>> state = new List<List<int>>();
                state.Add(new List<int>());
                state[0][0] = 2;
                state[0][1] = 1;

                return Tuple.Create(state, done);

            }

            public void remember(List<List<int>> state, int action, int reward, List<List<int>> next_state, bool done)
            {
                
                this.memory.PushRight((state, action, reward, next_state, done));
            }
        }
    }
}
