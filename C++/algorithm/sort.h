#include<iostream>
using namespace std;


//ð������
int bubbleSort(int arr[], int n)
{
	if (n == 0 || arr == NULL) {
		cout << "the array is NULL or no element." << endl;
		return 0;
	}

	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n - i - 1; j++) {
			if (arr[j + 1] < arr[j]) {
				//swap(arr[j], arr[j + 1]);
				int temp = arr[j];
				arr[j] = arr[j + 1];
				arr[j + 1] = temp;
			}
		}
	}

	return 0;
}


//ѡ������
int selectSort(int arr[], int n)
{
	if (n == 0 || arr == NULL) {
		cout << "the array is NULL or no element." << endl;
		return 0;
	}

	for (int i = 0; i < n; i++) {
		int idx = i;
		for (int j = i + 1; j < n; j++) {
			if (arr[idx] > arr[j]) {
				idx = j;
			}
		}
		if (i != idx) {
			int temp = arr[i];
			arr[i] = arr[idx];
			arr[idx] = temp;
		}
	}

	return 0;
}


//��������
int insertSort(int arr[], int n)
{
	if (n == 0 || arr == NULL) {
		cout << "the array is NULL or no element." << endl;
		return 0;
	}

	for (int i = 1; i < n; i++) {
		int key = arr[i];
		int j = i - 1;
		while (j >= 0 && arr[j]>key)
		{
			arr[j + 1] = arr[j];
			j--;
		}
		arr[j + 1] = key;
	}

	return 0;
}


//ϣ������
int shellSort(int arr[], int n)
{
	if (n == 0 || arr == NULL) {
		cout << "the array is NULL or no element." << endl;
		return 0;
	}

	//���ú��ʵ�gap
	int gap = 1;
	while (gap < n / 3)
	{
		gap = 3 * gap + 1;
	}

	//����
	while (gap >= 1)
	{
		for (int i = gap; i < n; i++) {
			for (int j = i; j >= gap && arr[j - gap]>arr[j]; j -= gap) {
				int temp = arr[j];
				arr[j] = arr[j - gap];
				arr[j - gap] = temp;
			}
		}
		gap /= 3;
	}

	return 0;
}


//�鲢����
//�ϲ���������
int merge(int arr[], int low, int mid, int high)
{

	int l = low, h = mid + 1;
	int *temp = new int[high - low + 1];
	int i = 0;
	while (l<=mid && h<=high)
	{
		*(temp + i++) = arr[l] <= arr[h] ? arr[l++] : arr[h++];
	}

	//��������ʣ���Ԫ�طŽ���ʱ����
	while (l<=mid)
	{
		*(temp + i++) = arr[l++];
	}
	while (h<=high)
	{
		*(temp + i++) = arr[h++];
	}

	//������Ż�����
	for (int j = 0; j < high - low + 1; j++) {
		arr[low + j] = *(temp + j);
	}

	delete[] temp;

	return 0;
}
//����
int mergeSort(int arr[], int low, int high)
{
	if (arr == NULL) {
		cout << "the array is NULL or no element." << endl;
		return 0;
	}

	if (low < high) {
		int mid = (low + high) / 2;
		mergeSort(arr, low, mid);
		mergeSort(arr, mid + 1, high);
		merge(arr, low, mid, high);
	}

	return 0;
}


//��������
int quickSort(int arr[], int low, int high)
{
	if (arr == NULL || low >= high)
		return 0;

	int key = arr[low];
	int l = low, h = high;
	while (l<h)
	{
		while (l < h && arr[h] >= key)
			h--;
		arr[l] = arr[h];
		while (l < h && arr[l] <= key)
			l++;
		arr[h] = arr[l];
	}

	arr[l] = key;
	quickSort(arr, low, l - 1);
	quickSort(arr, l + 1, high);
	
	return 0;
}


//������
//�ݹ鹹�������(len��arr�ĳ��ȣ�index�ǵ�һ����Ҷ�ӽڵ���±�)
int heapify(int arr[], int len, int index)
{
	int left = 2 * index + 1;  //index�����ӽڵ�
	int right = 2 * index + 2;  //index�����ӽڵ�
	int maxIdx = index;

	if (left < len && arr[left] > arr[maxIdx])
		maxIdx = left;
	if (right < len && arr[right] > arr[maxIdx])
		maxIdx = right;
	if (maxIdx != index) {
		swap(arr[maxIdx], arr[index]);
		heapify(arr, len, maxIdx);
	}

	return 0;
}
//����
int heapSort(int arr[], int n)
{
	//��������ѣ������һ����Ҷ�ӽڵ����ϣ�
	for (int i = n / 2 - 1; i >= 0; i--)
		heapify(arr, n, i);

	//���������
	for (int i = n - 1; i >= 1; i--) {
		swap(arr[0], arr[i]);  //����ǰ���ķ��õ�����ĩβ
		heapify(arr, i, 0);  //��δ�������Ĳ��ּ������ж�����
	}

	return 0;
}


//��������
//��������
int countSort(int arr[], int n)
{
	if (n == 0 || arr == NULL) {
		cout << "the array is NULL or no element." << endl;
		return 0;
	}

	//Ѱ�������е���С���ֵ
	int maxVal = arr[0];
	int minVal = arr[0];
	for (int i = 1; i < n; i++) {
		if (arr[i] > maxVal)
			maxVal = arr[i];
		if (arr[i] < minVal)
			minVal = arr[i];
	}

	//ȷ��ͳ�����鳤�Ȳ����г�ʼ��
	int size = maxVal - minVal + 1;
	int *countArr = new int[size];
	for (int i = 0; i < size; i++)
		countArr[i] = 0;
	for (int i = 0; i < n; i++) {
		countArr[arr[i]-minVal]++;
	}

	//����
	int idx = 0;
	for (int i = 0; i < size; i++) {
		if (countArr[i] > 0) {
			int j = countArr[i];
			while (j>0)
			{
				arr[idx] = i + minVal;
				idx++;
				j--;
			}
		}
	}

	delete[] countArr;

	return 0;
}

//��������
//�����ݵ����λ��,�����������
int maxBit(int arr[], int n)
{
	int bit = 1;
	int dec = 10;
	for (int i = 0; i < n; i++) {
		while (arr[i] >= dec)
		{
			dec *= 10;
			bit++;
		}
	}

	return bit;
}
//���򣨵�λ���ȣ�
int radixSort(int arr[], int n)
{
	if (n == 0 || arr == NULL) {
		cout << "the array is NULL or no element." << endl;
		return 0;
	}

	int times = maxBit(arr, n);  //����������λ�������������
	int radix = 1;  //����
	int bucket[10];  //������
	int *temp = new int[n];  //��ʱ����

	for (int i = 0; i < times; i++) {

		for (int j = 0; j < 10; j++)
			bucket[j] = 0;  //ÿ�η���ǰ��ռ�����

		for (int j = 0; j < n; j++) {
			int k = (arr[j] / radix) % 10;  //ͳ��ÿ��Ͱ�еļ�¼��
			bucket[k]++;
		}

		for (int j = 1; j < 10; j++)
			bucket[j] += bucket[j - 1];  //��temp�е�λ�����η����ÿ��Ͱ

	    //������Ͱ�м�¼�����ռ���temp��
		for (int j = n - 1; j >= 0; j--) {
			int k = (arr[j] / radix) % 10;
			temp[bucket[k] - 1] = arr[j];
			bucket[k]--;
		}

		//����ʱ��������ݸ��Ƶ�ԭ������
		for (int j = 0; j < n; j++)
			arr[j] = temp[j];

		radix *= 10;
	}

	delete[] temp;

	return 0;
}