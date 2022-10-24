using System.Collections.Generic;
using System.Diagnostics;
using Tensorflow.NumPy;
using Tensorflow;
using TorchSharp;
using static TorchSharp.torch.nn;
using static TorchSharp.TensorExtensionMethods;
using System.Collections;
using System.Linq;
using TorchSharp.Modules;
using System.Runtime.InteropServices;
using Tensorflow.Keras.Layers;

namespace RotinasIniciais
{
  class Program
  {
    static void Main(string[] args)
    {
      //var a = np.array(new int[,] { { 1, 2 }, { 5, 6 } });
      //var b = np.array(new int[,] { { 3, 4 }, { 7, 8 } });
      //a = np.concatenate(new NDArray[] { a, b });




      // Torch

      //var lin1 = Linear(1000, 100);
      //var lin2 = Linear(100, 10);
      //var seq = Sequential(("lin1", lin1), ("relu1", ReLU()), ("drop1", Dropout(0.1)), ("lin2", lin2));

      int input_size = 2;
      int output_size = 2;
      int neurons = 128;
      int layers = 5;
      
      var model = Sequential();
      model.append(Linear(input_size, neurons));
      for (int i = 0; i < layers; i++)
      {
        model.append(Linear(neurons, neurons));
        model.append(ReLU());
      }
      model.append(Linear(neurons, output_size));
      model.parameters();

      var seq = model;
      
      //var x = torch.randn(64, 1000);
      //var y = torch.randn(64, 10);

      //x.print();
      //Console.WriteLine(x[0][0].item<float>());
      //x[0][0].print();

      int size = 100;
      float[,] _x = new float[size, 2];
      for (int i = 0; i < size; i++)
      {
        for (int j = 0; j < 2; j++)
        {
          _x[i, j] = i+1;
        }
      }
      var x = torch.tensor(_x);
      x.print();


      float[,] _y = new float[size, 2];
      for (int i = 0; i < size; i++)
      {
        for (int j = 0; j < 2; j++)
        {
          _y[i, j] = (float)Math.Pow(i+1,j+1);
        }
      }

      var y = torch.tensor(_y);
      y.print();

      var optimizer = torch.optim.Adam(seq.parameters());

      for (int i = 0; i < 100; i++)
      {
        var eval = seq.forward(x);
        //eval.print();

        var output = functional.mse_loss(eval, y, torch.nn.Reduction.Sum);
        Console.WriteLine($"iter {i}");
        output.print();

        optimizer.zero_grad();

        output.backward();

        optimizer.step();
      }

      var rnd = new Random();
      var t = rnd.Next(size);
      var test = torch.tensor(new float[,] { {t, t} });
      test.print();
      var _test = seq.forward(test);
      _test.print();
      // End Torch

      int episodes = 1000;
      List<int> rewards = new();
      
      var dqn = new DQN_Keras(Convert.ToInt32(5 * Math.Pow(10, 5)), 32, 0.99, 1.0, 0.01, 0.999, 0.005, 0.125, 2, 2);
      (NDArray state, bool done) = dqn.reset_env();
      NDArray next_state = state;
      int reward = 0;
      int action = -1;

      for (int ep = 0; ep < episodes; ep++)
      {
        (state, done) = dqn.reset_env();
        List<int> r = new();
        while (!done)
        {
          action = dqn.get_action(state);
          (next_state, reward, done) = dqn.step(state, action);
          dqn.remember(state, action, reward, next_state, done);
          if (ep % 32 == 0)
            dqn.replay();
          state = next_state;
          r.Add(reward);
        }
        rewards.Add(r.Sum());
        // Console.WriteLine($"Next state: [{String.Join(",", state)}], Reward: {reward}, Done?: {done}");
        Console.WriteLine($"Episode {ep}, Reward: {rewards[ep]}");
      }

      

      //Console.WriteLine("oi");

      //var trainData = FileReaderMNIST.LoadImagesAndLables(
      //"./data/train-labels-idx1-ubyte.gz",
      //"./data/train-images-idx3-ubyte.gz");

      //var testData = FileReaderMNIST.LoadImagesAndLables(
      //    "./data/t10k-labels-idx1-ubyte.gz",
      //    "./data/t10k-images-idx3-ubyte.gz");



      //int[] layers = new int[4] { 784, 85, 170, 10 };
      //string[] activation = new string[3] { "relu", "relu", "softmax" };
      //NeuralNetwork net = new NeuralNetwork(layers, activation);
      //double[] resposta0 = { 1, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
      //double[] resposta1 = { 0, 1, 0, 0, 0, 0, 0, 0, 0, 0 };
      //double[] resposta2 = { 0, 0, 1, 0, 0, 0, 0, 0, 0, 0 };
      //double[] resposta3 = { 0, 0, 0, 1, 0, 0, 0, 0, 0, 0 };
      //double[] resposta4 = { 0, 0, 0, 0, 1, 0, 0, 0, 0, 0 };
      //double[] resposta5 = { 0, 0, 0, 0, 0, 1, 0, 0, 0, 0 };
      //double[] resposta6 = { 0, 0, 0, 0, 0, 0, 1, 0, 0, 0 };
      //double[] resposta7 = { 0, 0, 0, 0, 0, 0, 0, 1, 0, 0 };
      //double[] resposta8 = { 0, 0, 0, 0, 0, 0, 0, 0, 1, 0 };
      //double[] resposta9 = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 };

      //int contadorz = 0;
      //foreach (var image in trainData)
      //{
      //    Console.WriteLine(contadorz);
      //    contadorz++;
      //    if (image.Label == 0)
      //    {
      //        net.BackPropagate(byteArrToFloat(image.Image), resposta0);
      //    }
      //    else if (image.Label == 1)
      //    {
      //        net.BackPropagate(byteArrToFloat(image.Image), resposta1);
      //    }
      //    else if(image.Label == 2)
      //    {
      //        net.BackPropagate(byteArrToFloat(image.Image), resposta2);
      //    }
      //    else if (image.Label == 3)
      //    {
      //        net.BackPropagate(byteArrToFloat(image.Image), resposta3);
      //    }
      //    else if (image.Label == 4)
      //    {
      //        net.BackPropagate(byteArrToFloat(image.Image), resposta4);
      //    }
      //    else if (image.Label == 5)
      //    {
      //        net.BackPropagate(byteArrToFloat(image.Image), resposta5);
      //    }
      //    else if (image.Label == 6)
      //    {
      //        net.BackPropagate(byteArrToFloat(image.Image), resposta6);
      //    }
      //    else if (image.Label == 7)
      //    {
      //        net.BackPropagate(byteArrToFloat(image.Image), resposta7);
      //    }
      //    else if (image.Label == 8)
      //    {
      //        net.BackPropagate(byteArrToFloat(image.Image), resposta8);
      //    }
      //    else if (image.Label == 9)
      //    {
      //        net.BackPropagate(byteArrToFloat(image.Image), resposta9);
      //    }
      //}



      //int contador = 0;
      //foreach (var image in testData)
      //{
      //    if (contador > 1)
      //    {
      //        break;
      //    }
      //    contador++;
      //    double[] arrResposta = net.FeedForward(byteArrToFloat(image.Image));



      //    Console.WriteLine("Reposta verdadeira: ");
      //    Console.WriteLine(image.Label);

      //    Console.WriteLine("Reposta da rede: ");
      //    for (int i = 0; i < arrResposta.Length; i++)
      //    {
      //        Console.WriteLine(arrResposta[i]);
      //    }
      //}

      //contador = 0;
      //foreach (var image in trainData)
      //{
      //    if (contador > 1)
      //    {
      //        break;
      //    }
      //    contador++;
      //    double[] arrResposta = net.FeedForward(byteArrToFloat(image.Image));

      //    Console.WriteLine("Reposta verdadeira: ");
      //    Console.WriteLine(image.Label);


      //    Console.WriteLine("Reposta da rede: ");
      //    for (int i = 0; i < arrResposta.Length; i++)
      //    {
      //        Console.WriteLine(arrResposta[i]);
      //    }
      //}

      //int tamTeste = 10000;
      //float acc;
      //float acertos = 0;
      //foreach (var image in testData)
      //{
      //    double[] arrResposta = net.FeedForward(byteArrToFloat(image.Image));
      //    if (arrResposta.ToList().IndexOf(arrResposta.Max()) == image.Label)
      //    {
      //        acertos++;
      //    }

      //}

      //Console.WriteLine("Acurácia Teste: ");
      //Console.WriteLine(Convert.ToString(acertos/tamTeste));

      //int tamTreino = 60000;
      //float accTreino;
      //float acertosTreino = 0;
      //foreach (var image in trainData)
      //{
      //    double[] arrResposta = net.FeedForward(byteArrToFloat(image.Image));
      //    if (arrResposta.ToList().IndexOf(arrResposta.Max()) == image.Label)
      //    {
      //        acertosTreino++;
      //    }

      //}

      //Console.WriteLine("Acurácia Treino: ");
      //Console.WriteLine(Convert.ToString(acertosTreino/tamTreino));

      //Console.Write("Press any key to close the app...");
      //Console.ReadKey();


    }

    public static List<List<int>> CreateQTable()
    {
      // Por enquanto a Tabela Q é só uma lista de listas
      // Apagar depois porque é so fazer um new List[int[]] kkkkkk
      List<List<int>> qTable = new List<List<int>>();
      qTable.Add(new List<int>());
      qTable.Add(new List<int>());
      qTable.Add(new List<int>());
      qTable.Add(new List<int>());
      return qTable;
    }



    public static double[] byteArrToFloat(byte[,] byteArr)
    {
      int indice = 0;
      double[] imagem = new double[784];
      for (int i = 0; i < 28; i++)
      {
        for (int j = 0; j < 28; j++)
        {
          double d = byteArr[i, j] / 255.0;

          imagem[indice] = Convert.ToSingle(d);
          indice++;
        }
      }
      return imagem;
    }

  }
}