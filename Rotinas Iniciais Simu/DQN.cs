using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using DequeNet;

using Tensorflow;
using Tensorflow.Keras;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Engine;
using Tensorflow.NumPy;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;
using Tensorflow.Keras.Layers;
using Tensorflow.Keras.Models;

namespace RotinasIniciais
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
            private Deque<(List<int>, int, int, List<int>, bool)> memory;
            Random rnd = new Random();
            private Sequential model;


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
                this.memory = new Deque<(List<int>, int, int, List<int>, bool)>();
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
                this.model = CreateModel(2, 10, 10, 2);
        }

            public virtual (List<int>, int action, List<int>, int, bool) step(List<int> state, int action, bool random = false)
            {

                if (random) action = rnd.Next(2);
                List<int> next_state = new List<int>(state);

                if (action == 0)
                {
                    next_state[0] -= 1;
                    Console.WriteLine("andei pra esquerda");
                }
                else
                {
                    next_state[0] += 1;
                    Console.WriteLine("andei pra direita");
                }
                var done = false;
                if (next_state[0] == 0 || next_state[0] == 3)
                {
                    done = true;
                    Console.WriteLine("caiu no fogo ou ganhou");

                }
                if (next_state[1] == 1 && next_state[0] == 1)
                {
                    next_state[1] = 0;
                    Console.WriteLine("diamante parou de existir");
            }
                return (state, action, next_state, calc_reward(state, next_state, done), done);
            }

            public virtual int calc_reward(List<int> state, List<int> next_state, bool done)
            {
                if (next_state[0] == 0)
                {
                    return -10;
                }
                else if (next_state[0] == 3)
                {
                    return 10;
                }
                else if (state[1] == 1 && next_state[0] == 1)
                {
                    return 3;
                }
                else
                {
                    return 0;
                }
            }

            public virtual (List<int>, bool) reset_env()
            {
                var done = false;
                List<int> state = new List<int>();
                state.Add(2);
                state.Add(1);
                state[0] = 2;
                state[1] = 1;

                return (state, done);

            }

            public void remember(List<int> state, int action, int reward, List<int> next_state, bool done)
            {
                
                this.memory.PushRight((state, action, reward, next_state, done));
            }

            public void replay()
            {
                if(this.memory.Count < this.batch_size)  return;

                var samples = this.memory.OrderBy(x => rnd.Next()).Take(batch_size);
                
                
                foreach((var state, var action, var reward, var next_state, var done) in samples)
                {
                    var target = this.model.predict(tf.convert_to_tensor(state));
                    //if (done) target[0][action] = reward;
                    //else
                    //{
                        //Q_future = max(self.target_model(next_state).numpy()[0]);
                        //target[0][action] = reward + Q_future * self.gamma;
                    //}
                        
                }

            }

            static Sequential CreateModel(int layers, int neurons, int input_size, int output_size)
            {
                // Prepare layers
                var list_layers = new List<ILayer>();
                list_layers.Add(keras.layers.Dense(neurons, keras.activations.Relu, input_shape: new Shape(input_size)));
                for (int i = 0; i < layers; i++) list_layers.Add(keras.layers.Dense(64, keras.activations.Relu));
                list_layers.Add(keras.layers.Dense(output_size, keras.activations.Softmax));

                //Build sequential model
                var model = keras.Sequential(layers: list_layers);
                for (int i = 1, c = list_layers.Count; i < c; i++) model.Layers.Add(list_layers[i]);

                // Model Compile
                model.compile(optimizer: keras.optimizers.Adam(), loss: keras.losses.CategoricalCrossentropy(), metrics: new string[] { "accuracy" });

                return model;
            }
    }
    }

