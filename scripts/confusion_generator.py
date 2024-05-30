from cvss_types import ConfusionMatrixInfo


columns = ["A", "L"]
caption = "Generated Table"
label = "table:label"
row_labels = ["Row 1", "Row 2"]
data = [[1, 2], [3, 4]]


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
        output += " & " + f"{d:.2g}"

    output += " \\\\\n"
    return output


def rows(labels, data):
    output = ""
    for l, d in zip(labels, data):
        output += row(l, d)

    return output


def generate(info: ConfusionMatrixInfo):
    template = (
        """
    \\begin{table}
        \\caption{"""
        + f"{info['caption']}"
        + """}
        \\label{"""
        + f"{info['label']}"
        + """}
        \\begin{center}
            \\begin{tabular}{"""
        + f"{layout(len(info['columns']))}}}"
        + column_line(info["columns"])
        + """
                \\hline
        """
        + rows(info["row_labels"], info["data"])
        + """ 
            \\end{tabular}
        \\end{center}
    \\end{table}
    """
    )
    template.replace("_", "\\_")

    print(template)
