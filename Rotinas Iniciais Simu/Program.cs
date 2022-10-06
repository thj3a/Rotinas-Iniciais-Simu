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

            double num = 253/255.0;
            Console.WriteLine(num);

            float num2 = Convert.ToSingle(num);
            Console.WriteLine(num2);    

            // Ask the user to type the first number.
            Console.WriteLine("oi");

            var trainData = FileReaderMNIST.LoadImagesAndLables(
            "./data/train-labels-idx1-ubyte.gz",
            "./data/train-images-idx3-ubyte.gz");

            var testData = FileReaderMNIST.LoadImagesAndLables(
                "./data/t10k-labels-idx1-ubyte.gz",
                "./data/t10k-images-idx3-ubyte.gz");

           

            int[] layers = new int[4] { 784, 16, 16, 10 };
            string[] activation = new string[4] { "relu", "relu","relu", "softmax" };
            NeuralNetwork net = new NeuralNetwork(layers, activation);
            float[] resposta0 = { 1, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
            float[] resposta1 = { 0, 1, 0, 0, 0, 0, 0, 0, 0, 0 };
            float[] resposta2 = { 0, 0, 1, 0, 0, 0, 0, 0, 0, 0 };
            float[] resposta3 = { 0, 0, 0, 1, 0, 0, 0, 0, 0, 0 };
            float[] resposta4 = { 0, 0, 0, 0, 1, 0, 0, 0, 0, 0 };
            float[] resposta5 = { 0, 0, 0, 0, 0, 1, 0, 0, 0, 0 };
            float[] resposta6 = { 0, 0, 0, 0, 0, 0, 1, 0, 0, 0 };
            float[] resposta7 = { 0, 0, 0, 0, 0, 0, 0, 1, 0, 0 };
            float[] resposta8 = { 0, 0, 0, 0, 0, 0, 0, 0, 1, 0 };
            float[] resposta9 = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 };

            //int contadorzin = 1;
            foreach (var image in trainData)
            {

                if (image.Label == 0)
                {
                    net.BackPropagate(byteArrToFloat(image.Image), resposta0);
                }
                else if (image.Label == 1)
                {
                    net.BackPropagate(byteArrToFloat(image.Image), resposta1);
                }
                else if(image.Label == 2)
                {
                    net.BackPropagate(byteArrToFloat(image.Image), resposta2);
                }
                else if (image.Label == 3)
                {
                    net.BackPropagate(byteArrToFloat(image.Image), resposta3);
                }
                else if (image.Label == 4)
                {
                    net.BackPropagate(byteArrToFloat(image.Image), resposta4);
                }
                else if (image.Label == 5)
                {
                    net.BackPropagate(byteArrToFloat(image.Image), resposta5);
                }
                else if (image.Label == 6)
                {
                    net.BackPropagate(byteArrToFloat(image.Image), resposta6);
                }
                else if (image.Label == 7)
                {
                    net.BackPropagate(byteArrToFloat(image.Image), resposta7);
                }
                else if (image.Label == 8)
                {
                    net.BackPropagate(byteArrToFloat(image.Image), resposta8);
                }
                else if (image.Label == 9)
                {
                    net.BackPropagate(byteArrToFloat(image.Image), resposta9);
                }



               // contadorzin--;
                //if(contadorzin == 0)
                //{
                //    break;
                //}

                
            }

            //foreach(var image in data)
            //{


            //if (image.Label == 0)
            //{
            //float[] resposta = { 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
            // }
            // else if (image.Label == 1)
            // {
            //    float[] resposta = { 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
            // }
            // net.BackPropagate( image.Image , resposta );
            // //net.BackPropagate(new float[] { 1, 0, 0 }, new float[] { 1, 3 });
            // //net.BackPropagate(new float[] { 0, 1, 0 }, new float[] { 1, 4 });
            // //net.BackPropagate(new float[] { 0, 0, 1 }, new float[] { 1, 5 });
            // //net.BackPropagate(new float[] { 1, 1, 0 }, new float[] { 1, 6 });
            // //net.BackPropagate(new float[] { 0, 1, 1 }, new float[] { 1, 7 });
            // //net.BackPropagate(new float[] { 1, 0, 1 }, new float[] { 1, 8 });
            // //net.BackPropagate(new float[] { 1, 1, 1 }, new float[] { 1, 9 });
            //}

            //System.Threading.Thread.Sleep(4000);

            //Console.WriteLine(net.FeedForward(new float[] { 1 })[0]);
            //Console.WriteLine(net.FeedForward(new float[] { 2 })[0]);
            //Console.WriteLine(net.FeedForward(new float[] { 3 })[0]);
            //Console.WriteLine(net.FeedForward(new float[] { 4 })[0]);
            //Console.WriteLine(net.FeedForward(new float[] { 5 })[0]);
            //Console.WriteLine(net.FeedForward(new float[] { 6 })[0]);
            //Console.WriteLine(net.FeedForward(new float[] { 1 })[0]);

            int contador = 0;
            foreach(var image in testData)
            {
                if (contador > 1) {
                    break;
                }
                contador++;
                float[] arrResposta = net.FeedForward(byteArrToFloat(image.Image));

                

                Console.WriteLine("Reposta verdadeira: ");
                Console.WriteLine(image.Label);

                Console.WriteLine("Reposta da rede: ");
                for (int i = 0; i < 10; i++)
                {
                    Console.WriteLine(arrResposta[i]);
                }
            }

            contador = 0;
            foreach (var image in trainData)
            {
                if (contador > 1)
                {
                    break;
                }
                contador++;
                float[] arrResposta = net.FeedForward(byteArrToFloat(image.Image));

                Console.WriteLine("Reposta verdadeira: ");
                Console.WriteLine(image.Label);

                Console.WriteLine("o tamanho do bag");
                Console.WriteLine(arrResposta.Length);

                Console.WriteLine("Reposta da rede: ");
                for (int i = 0; i < 10; i++)
                {
                    Console.WriteLine(arrResposta[i]);
                }
            }

            //Console.WriteLine(net.FeedForward(byteArrToFloat(testData[0].Image)));





            Console.Write("Press any key to close the app...");
            Console.ReadKey();

        }
        
        public static List<List<int>> CreateQTable()
        {
            // Por enquanto a Tabela Q é só uma lista de listas
            List<List<int>> qTable = new List<List<int>>();
            qTable.Add(new List<int>());
            qTable.Add(new List<int>());
            qTable.Add(new List<int>());
            qTable.Add(new List<int>());
            return qTable;
        }

        

        public static float[] byteArrToFloat(byte[,] byteArr)
        {   
            int indice = 0;
            float[] imagem = new float[784];
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