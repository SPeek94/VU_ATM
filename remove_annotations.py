import xml.etree.ElementTree as ET
import glob

print(
    len(
        glob.glob(
            r"D:\Studie\Business Analytics\Applied Text Mining\workspace-negation-annotation\AI_BA_5_Sebastiaan_Peek\saved\*.txt.knowtator.xml"
        )
    )
)


# tree = ET.parse(
#     "workspace-negation-annotation\AI_BA_5_Sebastiaan_Peek\saved\ABC-News_20170511T203024.txt.knowtator.xml"
# )

# root = tree.getroot()


# for annotation in root.findall("annotation"):
#     annotator = annotation.find("annotator").text
#     mentionId = annotation.find("mention").get("id")
#     classMention_ann = root.find(".//classMention[@id = '%s']" % mentionId)
#     if annotator == "Stan":
#         root.remove(annotation)
#         root.remove(classMention_ann)


# for adjudicating in root.findall("adjudicating"):
#     annotator = adjudicating.find("annotator").text
#     mentionId = adjudicating.find("mention").get("id")
#     classMention_adj = root.find(".//classMention[@id = '%s']" % mentionId)
#     if annotator == "Stan":
#         root.remove(adjudicating)
#         root.remove(classMention_adj)

# tree.write("test.txt.knowtator.xml")

