#pragma once
#include <stdio.h>
#include <vector>
#include <string.h>
#include "NeuralLayer.h"
#include "NeuralNode.h"
#include "lib/libconvert.h"


//学习模式
typedef enum 
{
	Online,
	Batch,
	//输入向量如果0的项较多，在线学习会比较快
	//通常情况下批量学习会考虑全局优先，应为首选
	//在线学习每次都更新所有键结值，批量学习每一批数据更新一次键结值
} NeuralNetLearnMode;

//计算模式
typedef enum 
{
	ByLayer,
	ByNode,
} NeuralNetCalMode;

//工作模式
typedef enum
{
	Fit,  //拟合
	Classify,  //分类，会筛选最大值设为1，其他设为0
	Probability,   //几率，结果会归一化	
} NeuralNetWorkMode;


//神经网
class NeuralNet
{
public:
	NeuralNet();
	virtual ~NeuralNet();

	//神经层
	std::vector<NeuralLayer*> layers;
	std::vector<NeuralLayer*>& getLayerVector() { return layers; }

	std::vector<NeuralNode*> nodes;
	void initNodes();

	int id;

	NeuralLayer*& getLayer(int number) { return layers[number]; }
	NeuralLayer*& getFirstLayer() { return layers[0]; }
	NeuralLayer*& getLastLayer() { return layers[layers.size() - 1]; }
	int getLayerAmount() { return layers.size(); };

	int inputAmount;
	int outputAmount;
	int realDataAmount = 0;  //实际的数据量
	int nodeDataAmount = 0;  //节点的数据量

	NeuralNetLearnMode learnMode = Batch;

	double learnSpeed = 0.5;
	void setLearnSpeed(double s) { learnSpeed = s; }
	void setLearnMode(NeuralNetLearnMode lm);

	NeuralNetWorkMode workMode = Fit;
	void setWorkMode(NeuralNetWorkMode wm) { workMode = wm; }

	void createLayers(int amount);  //包含输入和输出层

	void learn(double* input, double* output, int amount);  //学习一组数据

	void train(int times = 1000000, double tol = 0.01);  //学习一批数据
	
	double calTol();

	void activeOutputValue(double* input, double* output, int amount);  //计算一组输出

	//数据
	double* inputData = nullptr;
	double* expectData = nullptr;
	void readData(const char* filename, double* input = nullptr, double* output = nullptr, int amount = -1);

	std::vector<bool> isTest;
	double* inputTestData = nullptr;
	double* expectTestData = nullptr;
	int testDataAmount = 0;
	void selectTest();
	void test();

	//具体设置
	virtual void createByData(NeuralLayerMode layerMode = HaveConstNode, int layerAmount = 3, int nodesPerLayer = 7); //具体的网络均改写这里
	void outputBondWeight(const char* filename = nullptr); //具体的网络均改写这里
	void createByLoad(const char* filename, bool haveConstNode = true);

	void setNodeDataAmount(int amount);

	NeuralNetCalMode activeMode = ByLayer;
	NeuralNetCalMode backPropageteMode = ByLayer;

};

