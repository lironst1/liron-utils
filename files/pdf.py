from PyPDF2 import PdfFileMerger


def merge_pdf(out, files):
    h = PdfFileMerger()
    [h.append(file) for file in files]
    h.write(out)
    h.close()
