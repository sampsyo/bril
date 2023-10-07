BEGIN {
    printf "\\begin{tabular}{|r|l|}\\hline\n\\multicolumn{2}{|c|}{%s}\\\\ \\hline\n", header
}
{
    if($0 != "")
	print $1 " & " $2 " \\\\ \\hline"
}
END {
    print "\\end{tabular}"
}
