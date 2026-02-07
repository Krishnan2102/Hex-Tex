import pymupdf

doc=pymupdf.open("Project Description.pdf")
out=open("output.txt","wb")
for page in doc:
    text=page.get_text().encode("utf8")
    out.write(text)
    out.write(bytes((12,)))
out.close()