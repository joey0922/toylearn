#include<iostream>
using namespace std;


//冒泡排序
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


//选择排序
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


//插入排序
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


//希尔排序
int shellSort(int arr[], int n)
{
	if (n == 0 || arr == NULL) {
		cout << "the array is NULL or no element." << endl;
		return 0;
	}

	//设置合适的gap
	int gap = 1;
	while (gap < n / 3)
	{
		gap = 3 * gap + 1;
	}

	//排序
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


//归并排序
//合并两个数组
int merge(int arr[], int low, int mid, int high)
{

	int l = low, h = mid + 1;
	int *temp = new int[high - low + 1];
	int i = 0;
	while (l<=mid && h<=high)
	{
		*(temp + i++) = arr[l] <= arr[h] ? arr[l++] : arr[h++];
	}

	//将数组中剩余的元素放进临时数组
	while (l<=mid)
	{
		*(temp + i++) = arr[l++];
	}
	while (h<=high)
	{
		*(temp + i++) = arr[h++];
	}

	//将结果放回数组
	for (int j = 0; j < high - low + 1; j++) {
		arr[low + j] = *(temp + j);
	}

	delete[] temp;

	return 0;
}
//排序
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


//快速排序
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


//堆排序
//递归构建大根堆(len是arr的长度，index是第一个非叶子节点的下标)
int heapify(int arr[], int len, int index)
{
	int left = 2 * index + 1;  //index的左子节点
	int right = 2 * index + 2;  //index的左子节点
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
//排序
int heapSort(int arr[], int n)
{
	//构建大根堆（从最后一个非叶子节点向上）
	for (int i = n / 2 - 1; i >= 0; i--)
		heapify(arr, n, i);

	//调整大根堆
	for (int i = n - 1; i >= 1; i--) {
		swap(arr[0], arr[i]);  //将当前最大的放置到数组末尾
		heapify(arr, i, 0);  //将未完成排序的部分继续进行堆排序
	}

	return 0;
}


//基数排序
//计数排序
int countSort(int arr[], int n)
{
	if (n == 0 || arr == NULL) {
		cout << "the array is NULL or no element." << endl;
		return 0;
	}

	//寻找数组中的最小最大值
	int maxVal = arr[0];
	int minVal = arr[0];
	for (int i = 1; i < n; i++) {
		if (arr[i] > maxVal)
			maxVal = arr[i];
		if (arr[i] < minVal)
			minVal = arr[i];
	}

	//确定统计数组长度并进行初始化
	int size = maxVal - minVal + 1;
	int *countArr = new int[size];
	for (int i = 0; i < size; i++)
		countArr[i] = 0;
	for (int i = 0; i < n; i++) {
		countArr[arr[i]-minVal]++;
	}

	//排序
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

//基数排序
//求数据的最大位数,决定排序次数
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
//排序（低位优先）
int radixSort(int arr[], int n)
{
	if (n == 0 || arr == NULL) {
		cout << "the array is NULL or no element." << endl;
		return 0;
	}

	int times = maxBit(arr, n);  //数列中最大的位数决定排序次数
	int radix = 1;  //基数
	int bucket[10];  //计数器
	int *temp = new int[n];  //临时数组

	for (int i = 0; i < times; i++) {

		for (int j = 0; j < 10; j++)
			bucket[j] = 0;  //每次分配前清空计数器

		for (int j = 0; j < n; j++) {
			int k = (arr[j] / radix) % 10;  //统计每个桶中的记录数
			bucket[k]++;
		}

		for (int j = 1; j < 10; j++)
			bucket[j] += bucket[j - 1];  //将temp中的位置依次分配给每个桶

	    //将所有桶中记录依次收集到temp中
		for (int j = n - 1; j >= 0; j--) {
			int k = (arr[j] / radix) % 10;
			temp[bucket[k] - 1] = arr[j];
			bucket[k]--;
		}

		//将临时数组的内容复制到原数组中
		for (int j = 0; j < n; j++)
			arr[j] = temp[j];

		radix *= 10;
	}

	delete[] temp;

	return 0;
}