#include<iostream>
using namespace std;
void q_sort(int *a,int begin,int end)
{
    int index_l = begin+1;
    int index_r = end;
    if(begin+1 > end)
        return;
    while(index_l < index_r)
    {
	while(a[index_l]<a[begin] && index_l<index_r)
	    index_l++;
	while(a[index_r]>a[begin] && index_r>index_l)
	    index_r--;
	if(index_r>index_l)
	{
	    int temp = a[index_r];
	    a[index_r] = a[index_l];
	    a[index_l] = temp;
	    index_r--;index_l++; 
	}	
    }
    if(a[begin]>a[index_l])
    {
	int temp = a[index_l];
        a[index_l] = a[begin];
        a[begin] = temp;
    }
    else
    {
	int temp = a[index_l - 1];
        a[index_l - 1] = a[begin];
        a[begin] = temp;

    }
    q_sort(a,begin,index_l);
    q_sort(a,index_l,end);
}
void putout(int *a, int len)
{
    for(int i = 0; i < len; i++)
        std::cout<<a[i]<<"\t";
}
int main()
{
    std::cout<<"Hello world"<<std::endl;
    int a[10] = {10,2,3,5,6,11,3,7,6,9};
    q_sort(a,0,9);
    putout(a,10);
    return 0;
}
int testmain()
{
	std::cout<<"Hello testmain"<<std::endl;
}
