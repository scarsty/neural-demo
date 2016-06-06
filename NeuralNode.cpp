#include "NeuralNode.h"
#include <cstdlib>

int NeuralNode::dataAmount = 0;

NeuralNode::NeuralNode()
{
	dataAmount = 0;
}


NeuralNode::~NeuralNode()
{
 	for (auto &b : prevBonds)
 	{
		delete b.second;
 	}
}

void NeuralNode::setExpect(double x, int i /*= -1*/)
{
	//this->expect = x;
	if (i >= 0 && i < dataAmount)
	{
		this->expects[i] = x;
	}
}

void NeuralNode::setInput(double x, int i /*= -1*/)
{
	//this->inputValue = x;
	if (i >= 0 && i < dataAmount)
	{
		this->inputValues[i] = x;
		if (type == Input)
			this->outputValues[i] = x;
	}
	if (type == Const) return;
}

void NeuralNode::setOutput(double x, int i /*= -1*/)
{
	if (type == Const) return;
	//this->outputValue = x;
	if (i >= 0 && i < dataAmount)
	{
		this->outputValues[i] = x;
	}
}

double NeuralNode::getOutput(int i /*= -1*/)
{
	//if (i < 0) return outputValue;
	//else 
	return outputValues[i];
}

//这里将多数据的情况写在了一起，可能需要调整
void NeuralNode::collectInputValue()
{
	//inputValue = 0;
	for (int i = 0; i < dataAmount; i++)
	{
		inputValues[i] = 0;
	}
	if (type == Const)
		return;
	for (auto &b : prevBonds)
	{
		//inputValue += b.second->startNode->outputValue * b.second->weight;
		for (int i = 0; i < dataAmount; i++)
		{
			inputValues[i] += b.second->startNode->outputValues[i] * b.second->weight;
		}
		//printf("\t%lf, %lf\n", b.second.startNode->outputValue, b.second.weight);
	}
	//printf("%lf\n",totalInputValue);
}

//同上
void NeuralNode::activeOutputValue()
{
	actived = true;
	if (type == Const)
	{
		//outputValue = -1;
		for (int i = 0; i < dataAmount; i++)
		{
			outputValues[i] = -1;
		}
		return;
	}
	if (type == Input) return;
	//outputValue = activeFunction(inputValue);
	for (int i = 0; i < dataAmount; i++)
	{
		outputValues[i] = activeFunction(inputValues[i]);
	}
}

void NeuralNode::active()
{
	collectInputValue();
	activeOutputValue();
}

void NeuralNode::setFunctions(std::function<double(double)> _active, std::function<double(double)> _dactive)
{
	activeFunction = _active;
	dactiveFunction = _dactive;
}

void NeuralNode::connect(NeuralNode* start, NeuralNode* end, double w /*= 0*/)
{
	if (w == 0)
	{
		w = 1.0 * rand() / RAND_MAX - 0.5;
	}
	auto bond = new NeuralBond();
	bond->startNode = start;
	bond->endNode = end;
	bond->weight = w;
	//这里内部维护两组连接，其中前连接为主，后连接主要用于计算delta
	//前连接
	end->prevBonds[start] = bond;
	//后连接
	start->nextBonds[end] = bond;
}

void NeuralNode::connectStart(NeuralNode* node, double w /*= 0*/)
{
	connect(node, this, w);
}

void NeuralNode::connectEnd(NeuralNode* node, double w /*= 0*/)
{
	connect(this, node, w);
}


void NeuralNode::updateOneDelta()
{
	/*
	delta = 0;
	if (this->type == Output)
	{
		delta = (expect - outputValue)*dactiveFunction(inputValue);
	}
	else
	{
		for (auto& b : nextBonds)
		{
			auto& bond = b.second;
			auto& node = bond->endNode;
			delta += node->delta*bond->weight;
		}
		delta = delta*dactiveFunction(inputValue);
	}
	*/
}


void NeuralNode::updateDelta()
{
	//this->updateOneDelta();
	for (int i = 0; i < dataAmount; i++)
	{
		deltas[i] = 0;
		if (this->type == Output)
		{
			deltas[i] = (expects[i] - outputValues[i]);
			//deltas[i] *= dactiveFunction(inputValues[i]);
			//这里如果去掉这个乘法，是使用交叉熵作为代价函数，但是在隐藏层的传播不可以去掉
		}
		else
		{
			for (auto& b : nextBonds)
			{
				auto& bond = b.second;
				auto& node = bond->endNode;
				deltas[i] += node->deltas[i] * bond->weight;
			}
			deltas[i] *= dactiveFunction(inputValues[i]);
		}
	}
}

//反向传播
void NeuralNode::backPropagate(double learnSpeed /*= 0.5*/)
{
	backPropageted = true;
	updateDelta();
	for (auto b : prevBonds)
	{
		auto& bond = b.second;
		bond->updateWeight(learnSpeed);
	}
}

void NeuralNode::setDataAmount(int n)
{
	dataAmount = n;
	inputValues.resize(n);
	outputValues.resize(n);
	expects.resize(n);
	deltas.resize(n);
	setVectorValue(outputValues);
}

void NeuralBond::updateWeight(double learnSpeed)
{
	auto& w = endNode->prevBonds[startNode]->weight;
	int n = startNode->dataAmount;

	double delta_w = 0;
	for (int i = 0; i < n; i++)
	{
		delta_w += endNode->deltas[i] * startNode->outputValues[i];
	}
	delta_w /= n;
	w += learnSpeed*delta_w;
}