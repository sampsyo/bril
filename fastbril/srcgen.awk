#!/usr/bin/env -S awk -f

{
    if($0 != "")
	print "#define " $1 " " $2;
}
