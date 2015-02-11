__kernel
void matrixSearch(__global char* MN, __global char* AB, __global char* RES,  int rowMN,  int colMN,  int rowAB,  int colAB )
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    
    if (((rowMN-i) > rowAB) && ((colMN-i) > colAB))
	{
		int match = 1;
		for ( int k = 0; k < rowAB; k++)
		{
			for ( int m = 0; m < colAB; m++)
			{
				if (MN[(i+k)*colMN+(j+m)] != AB[k*colMN+m])
				{
					match = 0;
					break;
				}
			}
			if (!match)
				break;
		}

		if (match)
		{
			RES[i*colMN+j] = 1;
		}
	}
}