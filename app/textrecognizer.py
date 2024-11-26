from PIL import Image
from tesserocr import PSM, RIL, iterate_level


def recognize(img, api):
    image = Image.fromarray(img)
    api.SetPageSegMode(PSM.AUTO)
    api.SetImage(image)
    api.Recognize()

    iterator = api.GetIterator()
    level = RIL.WORD

    result = list()
    for i, itr in enumerate(iterate_level(iterator, level)):
        try:
            word = itr.GetUTF8Text(level)
        except RuntimeError:
            continue

        confidence = itr.Confidence(level)

        if not word.strip() or confidence < 60:
            continue
        
        rect = itr.BoundingBox(level)
        result.append((i, word, rect))
    
    return result
