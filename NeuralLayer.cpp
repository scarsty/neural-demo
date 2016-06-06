#include "NeuralLayer.h"



NeuralLayer::NeuralLayer()
{
}


NeuralLayer::~NeuralLayer()
{
	for (auto& node : nodes)
	{
		delete node;
	}
}

//创建神经元
void NeuralLayer::createNodes(int nodeAmount, NeuralNodeType type /*= Hidden*/, NeuralLayerMode layerMode /*= HaveNotConstNode*/, int dataAmount /*= 0*/)
{
	//！！！注意实际上是多出了一个node，最后一个node的输出无论为何都恒定为1且不与前一层连接
	nodes.resize(nodeAmount);
	for (int i = 0; i < nodeAmount; i++)
	{
		auto node = new NeuralNode();
		node->type = type;
		node->id = i;
		node->setDataAmount(dataAmount);
		if (layerMode == HaveConstNode && i == nodeAmount - 1)
		{
			node->type = Const;
		}
		nodes[i] = node;
	}
}

//两个神经层所有节点都连接
void NeuralLayer::connetLayer(NeuralLayer* startLayer, NeuralLayer* endLayer)
{
	for (auto& startNode : startLayer->nodes)
	{
		for (auto& endNode : endLayer->nodes)
		{
			if (endNode->type != Const)
			{
				endNode->connectStart(startNode);
			}
		}
	}
}

void NeuralLayer::connetPrevlayer(NeuralLayer* prevLayer)
{
	connetLayer(prevLayer, this);
}

void NeuralLayer::connetNextlayer(NeuralLayer* nextLayer)
{
	connetLayer(this, nextLayer);
}

void NeuralLayer::markMax()
{
	if (getNodeAmount() <= 0) return;
	for (int i_data = 0; i_data < getNode(0)->getDataAmount(); i_data++)
	{
		double now_max = getNode(0)->getOutput(i_data);
		getNode(0)->setOutput(0, i_data);
		int pos = 0;
		for (int i_node = 1; i_node < getNodeAmount(); i_node++)
		{
			if (now_max <= getNode(i_node)->getOutput(i_data))
			{
				now_max = getNode(i_node)->getOutput(i_data);
				pos = i_node;
			}
			getNode(i_node)->setOutput(0, i_data);
		}
		getNode(pos)->setOutput(1, i_data);
	}
}

void NeuralLayer::normalized()
{
	for (int i = 0; i < NeuralNode::dataAmount; i++)
	{
		double sum = 0;
		for (auto& node : nodes)
		{
			sum += node->outputValues[i];
		}
		if (sum == 0) continue;
		for (auto& node : nodes)
		{
			node->outputValues[i] /= sum;
		}
	}
}
