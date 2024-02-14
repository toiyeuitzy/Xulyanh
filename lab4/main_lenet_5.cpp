#define _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <stdlib.h>
#include "data/tensor.h"
#include <math.h>

#define IMAGE_SIZE 28
#define KERNEL_SIZE_CONVOLUTION_1 1
#define DATA_LENGTH 20

void Convolution1(float input[28][28], float weight[6][1][1], float bias[6], float output[6][28][28])
{
    for (int channel = 0; channel < 6; channel++)
    {
        for (int row = 0; row < 28; row++)
        {
            for (int col = 0; col < 28; col++)
            {
                output[channel][row][col] = 0.0;
                for (int i = 0; i < 1; i++)
                {
                    for (int j = 0; j < 1; j++)
                    {
                        if (i == 0 && j == 0)
                        {
                            output[channel][row][col] = input[row + i][col + j] * weight[channel][i][j] + bias[channel];
                        }
                        else
                        {
                            output[channel][row][col] += input[row + i][col + j] * weight[channel][i][j];
                        }
                    }
                }
            }
        }
    }
}

void ReLU1(float input[6][28][28], float output[6][28][28])
{
    for (int channel = 0; channel < 6; channel++)
    {
        for (int row = 0; row < 28; row++)
        {
            for (int col = 0; col < 28; col++)
            {
                output[channel][row][col] = fmax(0, input[channel][row][col]);
            }
        }
    }
}

void ReLU2(float input[16][10][10], float output[16][10][10])
{
    for (int channel = 0; channel < 16; channel++)
    {
        for (int row = 0; row < 10; row++)
        {
            for (int col = 0; col < 10; col++)
            {
                output[channel][row][col] = fmax(0, input[channel][row][col]);
            }
        }
    }
}

void ReLU3(float input[120], float output[120])
{
    for (int col = 0; col < 120; col++)
    {
        output[col] = fmax(0, input[col]);
    }
}

void ReLU4(float input[84], float output[84])
{
    for (int col = 0; col < 84; col++)
    {
        output[col] = fmax(0, input[col]);
    }
}

void AveragePooling1(float input[6][28][28], float output[6][14][14])
{
    for (int channel = 0; channel < 6; channel++)
    {
        for (int row = 0; row < 14; row++)
        {
            for (int col = 0; col < 14; col++)
            {
                output[channel][row][col] = (input[channel][2 * row][2 * col] +
                                             input[channel][2 * row][2 * col + 1] +
                                             input[channel][2 * row + 1][2 * col] +
                                             input[channel][2 * row + 1][2 * col + 1]) /
                                            4.0;
            }
        }
    }
}

void AveragePooling2(float input[16][10][10], float output[16][5][5])
{
    for (int channel = 0; channel < 16; channel++)
    {
        for (int row = 0; row < 5; row++)
        {
            for (int col = 0; col < 5; col++)
            {
                output[channel][row][col] = (input[channel][2 * row][2 * col] +
                                             input[channel][2 * row][2 * col + 1] +
                                             input[channel][2 * row + 1][2 * col] +
                                             input[channel][2 * row + 1][2 * col + 1]) /
                                            4.0;
            }
        }
    }
}

void Convolution2(float input[6][14][14], float weight[16][6][5][5], float bias[16], float output[16][10][10])
{
    for (int channel = 0; channel < 16; channel++)
    {
        for (int row = 0; row < 10; row++)
        {
            for (int col = 0; col < 10; col++)
            {
                output[channel][row][col] = bias[channel];
                for (int k = 0; k < 5; k++)
                {
                    for (int i = 0; i < 5; i++)
                    {
                        for (int j = 0; j < 5; j++)
                        {
                            if (!(k == 0 && i == 0 && j == 0))
                            {
                                output[channel][row][col] += input[k][row + i][col + j] * weight[channel][k][i][j];
                            }
                        }
                    }
                }
            }
        }
    }
}

void FullyConnected1(float input[400], float weight[120][400], float bias[120], float output[120])
{
    for (int i = 0; i < 120; i++)
    {
        for (int j = 0; j < 400; j++)
        {
            if (j == 0)
            {
                output[i] = input[j] * weight[i][j] + bias[i];
            }
            else
                output[i] = output[i] + (input[j] * weight[i][j]);
        }
    }
}

void FullyConnected2(float input[120], float weight[84][120], float bias[84], float output[84])
{
    for (int i = 0; i < 84; i++)
    {
        for (int j = 0; j < 120; j++)
        {
            if (j == 0)
            {
                output[i] = input[j] * weight[i][j] + bias[i];
            }
            else
                output[i] = output[i] + (input[j] * weight[i][j]);
        }
    }
}

void FullyConnected3(float input[84], float weight[10][84], float bias[10], float output[10])
{
    for (int i = 0; i < 10; i++)
    {
        for (int j = 0; j < 84; j++)
        {
            if (j == 0)
            {
                output[i] = input[j] * weight[i][j] + bias[i];
            }
            else
                output[i] = output[i] + (input[j] * weight[i][j]);
        }
    }
}

void Flatten(float input[16][5][5], float output[400])
{
    for (int channel = 0; channel < 16; channel++)
    {
        int pos = channel * 5 * 5;
        for (int rows = 0; rows < 5; rows++)
        {
            int pos1 = rows * 5;
            for (int cols = 0; cols < 5; cols++)
            {
                int pos2 = pos + pos1 + cols;
                output[pos2] = input[channel][rows][cols];
            }
        }
    }
}

void Prediction(float image[28][28], float w_conv1[6][1][1], float w_conv2[16][6][5][5],
                float w_fc1[120][400], float w_fc2[84][120], float w_fc3[10][84],
                float b_conv1[6], float b_conv2[16], float b_fc1[120],
                float b_fc2[84], float b_fc3[10], float probs[10])
{
    float conv1_output[6][28][28];
    Convolution1(image, w_conv1, b_conv1, conv1_output);

    float after_ReLU1[6][28][28];
    ReLU1(conv1_output, after_ReLU1);

    float after_average_pooling[6][14][14];
    AveragePooling1(after_ReLU1, after_average_pooling);

    float conv2_output[16][10][10];
    Convolution2(after_average_pooling, w_conv2, b_conv2, conv2_output);

    float after_ReLU2[16][10][10];
    ReLU2(conv2_output, after_ReLU2);

    float after_average_pooling2[16][5][5];
    AveragePooling2(after_ReLU2, after_average_pooling2);

    float after_flatten[400];
    Flatten(after_average_pooling2, after_flatten);

    float after_fc1[120];
    FullyConnected1(after_flatten, w_fc1, b_fc1, after_fc1);

    float after_ReLU3[120];
    ReLU3(after_fc1, after_ReLU3);

    float after_fc2[84];
    FullyConnected2(after_ReLU3, w_fc2, b_fc2, after_fc2);

    float after_ReLU4[84];
    ReLU4(after_fc2, after_ReLU4);

    FullyConnected3(after_ReLU4, w_fc3, b_fc3, probs);
}

int main(int argc, char **argv)
{

    // float image[28][28];
    float w_conv1[6][1][1];
    float w_conv2[16][6][5][5];
    float w_fc1[120][400];
    float w_fc2[84][120];
    float w_fc3[10][84];
    float b_conv1[6];
    float b_conv2[16];
    float b_fc1[120];
    float b_fc2[84];
    float b_fc3[10];
    float probs[10];

    int i, j, m, n, index;
    FILE *fp;

    /* Load Weights from DDR->LMM */
    fp = fopen("data/weights/w_conv1.txt", "r");
    for (i = 0; i < 6; i++)
        fscanf(fp, "%f ", &(w_conv1[i][0][0]));
    fclose(fp);

    fp = fopen("data/weights/w_conv2.txt", "r");
    for (i = 0; i < 16; i++)
    {
        for (j = 0; j < 6; j++)
        {
            for (m = 0; m < 5; m++)
            {
                for (n = 0; n < 5; n++)
                {
                    index = 16 * i + 6 * j + 5 * m + 5 * n;
                    fscanf(fp, "%f ", &(w_conv2[i][j][m][n]));
                }
            }
        }
    }
    fclose(fp);

    fp = fopen("data/weights/w_fc1.txt", "r");
    for (i = 0; i < 120; i++)
    {
        for (j = 0; j < 400; j++)
            fscanf(fp, "%f ", &(w_fc1[i][j]));
    }
    fclose(fp);

    fp = fopen("data/weights/w_fc2.txt", "r");
    for (i = 0; i < 84; i++)
    {
        for (j = 0; j < 120; j++)
            fscanf(fp, "%f ", &(w_fc2[i][j]));
    }
    fclose(fp);

    fp = fopen("data/weights/w_fc3.txt", "r");
    for (i = 0; i < 10; i++)
    {
        for (j = 0; j < 84; j++)
            fscanf(fp, "%f ", &(w_fc3[i][j]));
    }
    fclose(fp);

    fp = fopen("data/weights/b_conv1.txt", "r");
    for (i = 0; i < 6; i++)
        fscanf(fp, "%f ", &(b_conv1[i]));
    fclose(fp);

    fp = fopen("data/weights/b_conv2.txt", "r");
    for (i = 0; i < 16; i++)
        fscanf(fp, "%f ", &(b_conv2[i]));
    fclose(fp);

    fp = fopen("data/weights/b_fc1.txt", "r");
    for (i = 0; i < 120; i++)
        fscanf(fp, "%f ", &(b_fc1[i]));
    fclose(fp);

    fp = fopen("data/weights/b_fc2.txt", "r");
    for (i = 0; i < 84; i++)
        fscanf(fp, "%f ", &(b_fc2[i]));
    fclose(fp);

    fp = fopen("data/weights/b_fc3.txt", "r");
    for (i = 0; i < 10; i++)
        fscanf(fp, "%f ", &(b_fc3[i]));
    fclose(fp);

    float *dataset = (float *)malloc(LABEL_LEN * 28 * 28 * sizeof(float));
    int target[LABEL_LEN];

    fp = fopen("data/mnist-test-target.txt", "r");
    for (i = 0; i < LABEL_LEN; i++)
        fscanf(fp, "%d ", &(target[i]));
    fclose(fp);

    fp = fopen("data/mnist-test-image.txt", "r");
    for (i = 0; i < LABEL_LEN * 28 * 28; i++)
        fscanf(fp, "%f ", &(dataset[i]));
    fclose(fp);

    float image[28][28];
    float *datain;
    int acc = 0;
    int mm, nn;
    for (i = 0; i < DATA_LENGTH; i++)
    {

        datain = &dataset[i * 28 * 28];
        for (mm = 0; mm < 28; mm++)
            for (nn = 0; nn < 28; nn++)
            {
                image[mm][nn] = *(float *)&datain[28 * mm + nn];
            }

        Prediction(image, w_conv1, w_conv2, w_fc1, w_fc2, w_fc3, b_conv1, b_conv2, b_fc1, b_fc2, b_fc3, probs);

        int index = 0;
        float max = probs[0];

        for (j = 1; j < 10; j++)
        {
            if (probs[j] > max)
            {
                index = j;
                max = probs[j];
            }
        }
        if (index == target[i])
            acc++;
        printf("Predicted label: %d\n", index);
        printf("Prediction: %d/%d\n", acc, i + 1);
    }
    printf("Accuracy = %f\n", acc * 1.0f / DATA_LENGTH);

    return 0;
}