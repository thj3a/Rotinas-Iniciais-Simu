// See https://aka.ms/new-console-template for more information
using System.Collections.Generic;
using Accord.Math;
using MNIST.IO;

namespace RotinasIniciais
{
    class Program
    {
        static void Main(string[] args)
        {


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

            var dqn = new DQN(Convert.ToInt32(5*Math.Pow(10,5)),32, 0.99, 1.0, 0.01, 0.999, 0.005, 0.125, 2, 2);
            (List<int> state, bool done) = dqn.reset_env();

            int reward = 0;

            for(int i = 0; i < 100; i++)
            {
                (state, int action, var next_state, reward, done) = dqn.step(state, 2, true);
                dqn.remember(state, action, reward, next_state, done);
                dqn.replay();
            }

            Console.WriteLine($"Prox estado: [{String.Join(",", state)}], Recompensa: {reward}, Cabou?: {done}");

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
                    double d = byteArr[i, j]/255.0;

                    imagem[indice] = Convert.ToSingle(d);
                    indice++;
                }
            }
            return imagem;
        }

    }
}