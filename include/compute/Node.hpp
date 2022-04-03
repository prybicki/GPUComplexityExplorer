#pragma once

struct Node
{
	struct Args
	{
		int a;
		float b;
	};
	Node(Args){}
};

Node a {{.a=0, .b=3.14f}};
