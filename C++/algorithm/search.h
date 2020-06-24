#include<iostream>


//���ֲ���
int binarySearch(int arr[], int len, int target)
{
	if (arr == NULL || len == 0) {
		std::cout << "array is empty!" << std::endl;
		return -1;
	}

	int low = 0, high = len - 1;
	while (low <= high)
	{
		int mid = (low + high) / 2;
		if (arr[mid]>target)
			high = mid - 1;
		else if (arr[mid] < target)
			low = mid + 1;
		else
			return mid;
	}

	return -1;
}


//��ֵ����
int interpolateSearch(int arr[], int len, int target)
{
	if (arr == NULL || len == 0) {
		std::cout << "array is empty!" << std::endl;
		return -1;
	}

	int low = 0, high = len - 1;
	while (low <= high)
	{
		int mid = low + (high - low) * (target - arr[low]) / (arr[high] - arr[low]);
		if (arr[mid]>target)
			high = mid - 1;
		else if (arr[mid] < target)
			low = mid + 1;
		else
			return mid;
	}

	return -1;
}


//쳲���������
int fibSize(int n)
{

	int size = 1;
	int a = 0, b = 1;
	while (b < n)
	{
		int temp = b;
		b = a + b;
		a = temp;
		size++;
	}

	return size;
}


int fibonacciSearch(int arr[], int len, int target)
{
	if (arr == NULL || len == 0) {
		std::cout << "array is empty!" << std::endl;
		return -1;
	}

	//��ʼ��쳲���������,ʹ���������һ��Ԫ��ֵ��С�պô��ڵ���ԭ���г���ֵ
	int size = fibSize(len);
	int *fib = new int[size];
	for (int i = 0; i < size; i++) {
		if (i == 0 || i == 1)
			fib[i] = 1;
		else
			fib[i] = fib[i - 1] + fib[i - 2];
	}


	//���ԭ���鳤��С��쳲������������һ��Ԫ��ֵ��С����ԭ�������һ�������䵽Ԫ��ֵ��С
	int *temp = new int[fib[size - 1]];
	for (int i = 0; i < fib[size - 1]; i++) {
		if (i < len)
			temp[i] = arr[i];
		else
			temp[i] = arr[len - 1];
	}

	int low = 0, high = fib[size - 1] - 1;
	int k = size - 1;
	int mid;
	while (low <= high)
	{
		if (k > 0)
			mid = low + fib[k - 1] - 1;
		else
			mid = low;

		if (target < temp[mid]) {
			high = mid - 1;
			k -= 1;
		}
		else if (target > temp[mid]) {
			low = mid + 1;
			k -= 2;
		}
		else {
			return mid < len ? mid : len;
		}
	}

	delete[] fib;
	delete[] temp;

	return -1;
}