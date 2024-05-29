columns = ["A", "L"]


def column_line(columns):
    bf = """ & \\textbf{"""
    output = "\\textbf{}"
    for col in columns:
        output += bf + f"{col}" + "}"

    return output + "\\\\"


def layout(num_cols):
    output = ""
    for i in range(num_cols + 1):
        if i == num_cols:
            output += "l"
            continue
        output += "l|"
    return output


def row(label, line):
    output = label
    for d in line:
        output += " & " + f"{d}"

    output += " \\\\\n"
    return output


def rows(labels, data):
    output = ""
    for l, d in zip(labels, data):
        output += row(l, d)

    return output


caption = "Generated Table"
# column_line = """\\textbf{} & \\textbf{Column 1} & \\textbf{Column 2} \\\\ """
label = "table:label"
# layout = ""
labels = ["Row 1", "Row 2"]
data = [[1, 2], [3, 4]]
template = (
    """
\\begin{table}
    \\caption{"""
    + f"{caption}"
    + """}
    \\label{"""
    + f"{label}"
    + """}
    \\begin{center}
        \\begin{tabular}{"""
    + f"{layout(len(columns))}}}"
    + column_line(columns)
    + """
            \\hline
    """
    + rows(labels, data)
    + """ 
        \\end{tabular}
    \\end{center}
\\end{table}
"""
)
print(template)
