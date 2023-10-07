#!/bin/awk -f

{
    if($0 != "")
	print "#define " $1 " " $2;
}
