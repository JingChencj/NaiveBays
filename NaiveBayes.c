#include<stdio.h>
#include<cstdlib>
#include<string.h>

//label数量和特征数量的全局变量
int label_num = 10;
int feature_num = 784;

double priotProb[10] = {0.0};//先验概率
double condProb[10][784][2] = {0.0};//条件概率
//文件读取
int trainsLabelNum[36000] = {0};//训练Label的数量
int predictLabelNum[6000] = {0};//测试Label的数量
int trainsFeatureNum[36000][785] = {0};//训练feature的数量。加一列label的值。
int predictFeatureNum[6000][784] = {0};//测试feature的数量

//预测结果数组，其中按照顺序存放预测到的label，因此数组的长度应该是测试集的长度
int ResultLabel[6000] = {0};

//Function declaration
void Train();
void Predict();
double CalcuProb(int img[], int label);
double CalcuAccu();

char p[10000];

//主函数
int main(int argc, char* argv[]){
	memset(trainsLabelNum,0,sizeof(trainsLabelNum));

	//读二值化文件
	FILE *fp;  
	if((fp=fopen("binary.txt","r"))==NULL) {  
	    printf("File cannot be opened\n");  
	    exit(1);  
	}

	printf("read successfully and start dividing\n");

	fgets(p,10000,fp);
	// for (int i = 0; i < 3; ++i)
	// {
	// 	fscanf(fp,"%d",&trainsLabelNum[i]);
	// 	printf("%d\n", trainsLabelNum[i]);
	// 	for (int j = 0; j < 5; ++j)
	// 	{
	// 		fscanf(fp,"%d",&trainsFeatureNum[i][j]);
	// 	}
	// }

	// for (int i = 3; i < 8; ++i)
	// {
	// 	fscanf(fp,"%d",&predictLabelNum[i]);
	// 	for (int j = 0; j < 5; ++j)
	// 	{
	// 		fscanf(fp,"%d",&predictFeatureNum[i][j]);
	// 	}
	// }

	for (int i = 0; i < 36000; ++i)
	{
		fscanf(fp,"%d",&trainsLabelNum[i]);
		for (int j = 0; j < 784; ++j)
		{
			fscanf(fp,"%d",&trainsFeatureNum[i][j]);
		}
	}

	for (int i = 0; i < 6000; ++i)
	{
		fscanf(fp,"%d",&predictLabelNum[i]);
		for (int j = 0; j < 784; ++j)
		{
			fscanf(fp,"%d",&predictFeatureNum[i][j]);
		}
	}		

	printf("dividing successfully\n");

	fclose(fp);

	//trainning
	/*
	Input: trainning label array ; trainning feature array
	Output: trainning label prob array; trainning feature condition prob array
	*/
	printf("Start trainning\n");
	Train();
	printf("Trainning cost\n");

	//Predicting
	/*
	Input: predicting feature array 
	Output: predicted label array
	*/
	printf("Start predicting\n");
	Predict();
	printf("Predicting cost\n");

	//predict accuracy
	/*
	compare predicted label array and initial predicting label array
	compute accuracy
	*/
	double accuracy = CalcuAccu();
	printf("预测精度为：%f\n", accuracy);

	return 0;
}

//训练
void Train(){
	int label;
	//计算先验概率和条件概率
	/*频数统计
	1.统计训练集的label数组中属于0-9的分别有多少个
	2.统计训练集的FEATURE数组中	
	*/
	for (int i = 0; i < 36000; ++i)
	{
		label = trainsLabelNum[i];

		priotProb[label] += 1;

		for (int j = 0; j < 784; ++j)
		{
			int temp = trainsFeatureNum[label][j];
			condProb[label][j][temp] += 1;
		}
	}

	//将概率归到[1.10001]
	for (int i = 0; i < 10; ++i)
	{
		for (int j = 0; j < 784; ++j)
		{
			//经过二值化之后图像只有0，1两种取值
			int pix_0 = condProb[i][j][0];
			int pix_1 = condProb[i][j][1];
	
			//计算0,1像素点对应的条件概率
			double prob_0 = (float(pix_0)/float(pix_0+pix_1))*1000000+1;
			double prob_1 = (float(pix_1)/float(pix_0+pix_1))*1000000+1;

			condProb[i][j][0] = prob_0;
			condProb[i][j][1] = prob_1;
		}
	}

}

//预测
void Predict(){
	for (int i = 0; i < 6000; ++i)
	{
		int max_label = 0;
		double max_prob = CalcuProb(predictFeatureNum[i],0);

		for (int j = 1; j < 10; ++j)
		{
			double prob = CalcuProb(predictFeatureNum[i],j);

			if (max_prob < prob)
			{
				max_label = j;
				max_prob = prob;
			}
		}

		ResultLabel[i] = max_label;
	}
}
	
//计算概率
double CalcuProb(int img[], int label){
	int prob = int(priotProb[label]);

	for (int i = 0; i < 784; ++i)
	{
		prob *= int(condProb[label][i][img[i]]);
	}

	return prob;
}

//计算预测精度
double CalcuAccu(){
	int count = 0;
	for (int i = 0; i < 6000; ++i)
	{
		if (predictLabelNum[i] == ResultLabel[i])
		{
			count++;
		}
	}
	printf("count:%d\n",count);

	return double(count)/6000;
}