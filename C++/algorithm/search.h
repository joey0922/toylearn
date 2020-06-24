#include<iostream>


//二分查找
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


//插值查找
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


//斐波那契查找
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

	//初始化斐波那契数列,使得数列最后一个元素值大小刚好大于等于原数列长度值
	int size = fibSize(len);
	int *fib = new int[size];
	for (int i = 0; i < size; i++) {
		if (i == 0 || i == 1)
			fib[i] = 1;
		else
			fib[i] = fib[i - 1] + fib[i - 2];
	}


	//如果原数组长度小于斐波那契数组最后一个元素值大小，用原数组最后一个数补充到元素值大小
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