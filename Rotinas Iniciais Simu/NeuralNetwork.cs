using System;
using System.Collections.Generic;
using System.IO;

public class NeuralNetwork //: MonoBehaviour
{
  //fundamental
  private int[] layers;//layers

  private double[][] neurons;//neurons
  private double[][] biases;//biasses
  private double[][][] weights;//weights
  private int[] activations;//layers

  //genetic
  public double fitness = 0;//fitness

  //backprop
  public double learningRate = 0.001f;//learning rate

  public double cost = 0;

  private double[][] deltaBiases;//biasses
  private double[][][] deltaWeights;//weights
  private int deltaCount;

  public Random RNG = new Random();

  public NeuralNetwork(int[] layers, string[] layerActivations)
  {
    this.layers = new int[layers.Length];
    for (int i = 0; i < layers.Length; i++)
    {
      this.layers[i] = layers[i];
    }
    activations = new int[layers.Length - 1];
    for (int i = 0; i < layers.Length - 1; i++)
    {
      string action = layerActivations[i];
      switch (action)
      {
        case "sigmoid":
          activations[i] = 0;
          break;

        case "tanh":
          activations[i] = 1;
          break;

        case "relu":
          activations[i] = 2;
          break;

        case "leakyrelu":
          activations[i] = 3;
          break;

        case "softmax":
          activations[i] = 4;
          break;

        default:
          activations[i] = 2;
          break;
      }
    }
    InitNeurons();
    InitBiases();
    InitWeights();
  }

  private void InitNeurons()//create empty storage array for the neurons in the network.
  {
    List<double[]> neuronsList = new List<double[]>();
    for (int i = 0; i < layers.Length; i++)
    {
      neuronsList.Add(new double[layers[i]]);
    }
    neurons = neuronsList.ToArray();
  }

  private void InitBiases()//initializes random array for the biases being held within the network.
  {
    List<double[]> biasList = new List<double[]>();
    for (int i = 0; i < layers.Length; i++)
    {
      double[] bias = new double[layers[i]];
      for (int j = 0; j < layers[i]; j++)
      {
        bias[j] = (double)RNG.NextDouble() - 0.5d; ; // UnityEngine.Random.Range(-0.5f, 0.5f);
      }
      biasList.Add(bias);
    }
    biases = biasList.ToArray();
  }

  private void InitWeights()//initializes random array for the weights being held in the network.
  {
    List<double[][]> weightsList = new List<double[][]>();
    for (int i = 1; i < layers.Length; i++)
    {
      List<double[]> layerWeightsList = new List<double[]>();
      int neuronsInPreviousLayer = layers[i - 1];
      for (int j = 0; j < neurons[i].Length; j++)
      {
        double[] neuronWeights = new double[neuronsInPreviousLayer];
        for (int k = 0; k < neuronsInPreviousLayer; k++)
        {
          neuronWeights[k] = (double)RNG.NextDouble() - 0.5d; //UnityEngine.Random.Range(-0.5f, 0.5f);
        }
        layerWeightsList.Add(neuronWeights);
      }
      weightsList.Add(layerWeightsList.ToArray());
    }
    weights = weightsList.ToArray();
  }

  public double[] FeedForward(double[] inputs)//feed forward, inputs >==> outputs.
  {
    for (int i = 0; i < inputs.Length; i++)
    {
      neurons[0][i] = inputs[i];
    }
    for (int i = 1; i < layers.Length; i++)
    {
      double[] valueVec = new double[neurons[i].Length];
      int layer = i;
      for (int j = 0; j < neurons[i].Length; j++)
      {
        double value = 0.0;
        for (int k = 0; k < neurons[i - 1].Length; k++)
        {
          value += weights[i - 1][j][k] * neurons[i - 1][k];

        }


        if (activations[layer - 1] != 4)
        {
          neurons[i][j] = activate(value + biases[i][j], layer);
        }
        else
        {
          valueVec[j] = value;
        }
      }

      if (layer != 0)
      {
        if (activations[layer - 1] == 4)
        {
          double[] result = softmax(valueVec);
          for (int a = 0; a < neurons[i].Length; a++)
          {
            neurons[i][a] = result[a];
          }
        }
      }

    }
    return neurons[neurons.Length - 1];
  }

  //Backpropagation implemtation down until mutation.
  public double activate(double value, int layer)//all activation functions
  {
    switch (activations[layer])
    {
      case 0:
        return sigmoid(value);

      case 1:
        return tanh(value);

      case 2:
        return relu(value);

      case 3:
        return leakyrelu(value);

      default:
        return relu(value);
    }
  }

  public double[] softmax(double[] valueVec)
  {
    double[] softmax = new double[valueVec.Length];
    double maxValue = valueVec.Average();



    double[] x_exp = GetExpArr(GetArrMinus(valueVec, maxValue));
    double x_exp_sum = x_exp.Sum();

    for (int i = 0; i < softmax.Length; i++)
    {
      softmax[i] = x_exp[i] / x_exp_sum;
    }

    return softmax;
  }

  public double activateDer(double value, int layer)//all activation function derivatives
  {
    switch (activations[layer])
    {
      case 0:
        return sigmoidDer(value);

      case 1:
        return tanhDer(value);

      case 2:
        return reluDer(value);

      case 3:
        return leakyreluDer(value);

      default:
        return reluDer(value);
    }
  }

  public double sigmoid(double x)//activation functions and their corrosponding derivatives
  {
    double k = (double)Math.Exp(x);
    return k / (1.0f + k);
  }

  public double tanh(double x)
  {
    return (double)Math.Tanh(x);
  }

  public double relu(double x)
  {
    return (0 >= x) ? 0 : x;
  }

  public double leakyrelu(double x)
  {
    return (0 >= x) ? 0.01f * x : x;
  }



  public double sigmoidDer(double x)
  {
    return x * (1 - x);
  }

  public double tanhDer(double x)
  {
    return 1 - (x * x);
  }

  public double reluDer(double x)
  {
    return (0 >= x) ? 0 : 1;
  }

  public double leakyreluDer(double x)
  {
    return (0 >= x) ? 0.01f : 1;
  }

  public void BackPropagate(double[] inputs, double[] expected)//backpropogation;
  {
    double[] output = FeedForward(inputs);//runs feed forward to ensure neurons are populated correctly

    cost = 0;
    for (int i = 0; i < output.Length; i++) cost += (double)Math.Pow(output[i] - expected[i], 2);//calculated cost of network
    cost = cost / 2;//this value is not used in calculions, rather used to identify the performance of the network

    double[][] gamma;

    List<double[]> gammaList = new List<double[]>();
    for (int i = 0; i < layers.Length; i++)
    {
      gammaList.Add(new double[layers[i]]);
    }
    gamma = gammaList.ToArray();//gamma initialization

    int layer = layers.Length - 2;
    for (int i = 0; i < output.Length; i++) gamma[layers.Length - 1][i] = (output[i] - expected[i]) * activateDer(output[i], layer);//Gamma calculation
    for (int i = 0; i < neurons[layers.Length - 1].Length; i++)//calculates the w' and b' for the last layer in the network
    {
      biases[layers.Length - 1][i] -= gamma[layers.Length - 1][i] * learningRate;
      for (int j = 0; j < neurons[layers.Length - 2].Length; j++)
      {
        weights[layers.Length - 2][i][j] -= gamma[layers.Length - 1][i] * neurons[layers.Length - 2][j] * learningRate;//*learning
      }
    }

    for (int i = layers.Length - 2; i > 0; i--)//runs on all hidden layers
    {
      layer = i - 1;
      for (int j = 0; j < neurons[i].Length; j++)//outputs
      {
        gamma[i][j] = 0;
        for (int k = 0; k < gamma[i + 1].Length; k++)
        {
          gamma[i][j] = gamma[i + 1][k] * weights[i][k][j];
        }
        gamma[i][j] *= activateDer(neurons[i][j], layer);//calculate gamma
      }
      for (int j = 0; j < neurons[i].Length; j++)//itterate over outputs of layer
      {
        biases[i][j] -= gamma[i][j] * learningRate;//modify biases of network
        for (int k = 0; k < neurons[i - 1].Length; k++)//itterate over inputs to layer
        {
          weights[i - 1][j][k] -= gamma[i][j] * neurons[i - 1][k] * learningRate;//modify weights of network
        }
      }
    }
  }

  //Genetic implementations down onwards until save.

  public void Mutate(int high, double val)//used as a simple mutation function for any genetic implementations.
  {
    for (int i = 0; i < biases.Length; i++)
    {
      for (int j = 0; j < biases[i].Length; j++)
      {
        // biases[i][j] = (UnityEngine.Random.Range(0f, high) <= 2) ? biases[i][j] += UnityEngine.Random.Range(-val, val) : biases[i][j];
        biases[i][j] = (high * (double)RNG.NextDouble()) <= 2 ? biases[i][j] += (val * (2 * ((double)RNG.NextDouble() - 0.5f))) : biases[i][j];
      }
    }

    for (int i = 0; i < weights.Length; i++)
    {
      for (int j = 0; j < weights[i].Length; j++)
      {
        for (int k = 0; k < weights[i][j].Length; k++)
        {
          // weights[i][j][k] = (UnityEngine.Random.Range(0f, high) <= 2) ? weights[i][j][k] += UnityEngine.Random.Range(-val, val) : weights[i][j][k];
          weights[i][j][k] = (RNG.NextDouble() * high) <= 2 ? weights[i][j][k] += (val * (2 * ((double)RNG.NextDouble() - 0.5f))) : weights[i][j][k];
        }
      }
    }
  }

  public int CompareTo(NeuralNetwork other) //Comparing For Genetic implementations. Used for sorting based on the fitness of the network
  {
    if (other == null) return 1;

    if (fitness > other.fitness)
      return 1;
    else if (fitness < other.fitness)
      return -1;
    else
      return 0;
  }

  public NeuralNetwork copy(NeuralNetwork nn) //For creatinga deep copy, to ensure arrays are serialzed.
  {
    for (int i = 0; i < biases.Length; i++)
    {
      for (int j = 0; j < biases[i].Length; j++)
      {
        nn.biases[i][j] = biases[i][j];
      }
    }
    for (int i = 0; i < weights.Length; i++)
    {
      for (int j = 0; j < weights[i].Length; j++)
      {
        for (int k = 0; k < weights[i][j].Length; k++)
        {
          nn.weights[i][j][k] = weights[i][j][k];
        }
      }
    }
    return nn;
  }

  //save and load functions
  public void Load(string path)//this loads the biases and weights from within a file into the neural network.
  {
    TextReader tr = new StreamReader(path);
    int NumberOfLines = (int)new FileInfo(path).Length;
    string[] ListLines = new string[NumberOfLines];
    int index = 1;
    for (int i = 1; i < NumberOfLines; i++)
    {
      ListLines[i] = tr.ReadLine();
    }
    tr.Close();
    if (new FileInfo(path).Length > 0)
    {
      for (int i = 0; i < biases.Length; i++)
      {
        for (int j = 0; j < biases[i].Length; j++)
        {
          biases[i][j] = double.Parse(ListLines[index]);
          index++;
        }
      }

      for (int i = 0; i < weights.Length; i++)
      {
        for (int j = 0; j < weights[i].Length; j++)
        {
          for (int k = 0; k < weights[i][j].Length; k++)
          {
            weights[i][j][k] = double.Parse(ListLines[index]); ;
            index++;
          }
        }
      }
    }
  }

  public void Save(string path)//this is used for saving the biases and weights within the network to a file.
  {
    File.Create(path).Close();
    StreamWriter writer = new StreamWriter(path, true);

    for (int i = 0; i < biases.Length; i++)
    {
      for (int j = 0; j < biases[i].Length; j++)
      {
        writer.WriteLine(biases[i][j]);
      }
    }

    for (int i = 0; i < weights.Length; i++)
    {
      for (int j = 0; j < weights[i].Length; j++)
      {
        for (int k = 0; k < weights[i][j].Length; k++)
        {
          writer.WriteLine(weights[i][j][k]);
        }
      }
    }
    writer.Close();
  }

  public double[] GetExpArr(double[] arr)
  {
    if (double.IsNaN(arr[0]))
    {
      Console.WriteLine("O BAGULHO ENTROU ERRADOO");
    }
    double num;
    double[] expArr = new double[arr.Length];
    for (int i = 0; i < arr.Length; i++)
    {
      num = Math.Exp(arr[i]);



      expArr[i] = num;
    }
    return expArr;
  }

  public double[] GetArrMinus(double[] arr, double num)
  {
    double[] arrMinusNum = new double[arr.Length];
    for (int i = 0; i < arr.Length; i++)
    {
      arrMinusNum[i] = arr[i] - num;
    }
    return arrMinusNum;
  }

}
