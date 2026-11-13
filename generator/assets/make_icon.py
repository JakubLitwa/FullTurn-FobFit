"""Draw the FobFit icon: a car key head inside an orange FobFit jacket with its wing, blade up.

  python assets/make_icon.py        writes assets/fobfit.ico (16-256 px) and assets/fobfit.png (256 px)

Each size is drawn on its own (4x supersampled), so the small sizes stay sharp: below 32 px the buttons
and the ring hole are left out and the blade and rim get wider.
"""
from pathlib import Path

from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
SIZES = (16, 20, 24, 32, 40, 48, 64, 128, 256)

ORANGE, ORANGE_DARK, ORANGE_LIGHT = (242, 113, 28), (150, 62, 8), (255, 168, 92)
HEAD, HEAD_EDGE, BUTTON, BUTTON_EDGE = (38, 43, 51), (12, 15, 19), (78, 87, 100), (20, 24, 30)
BLADE, BLADE_EDGE, BLADE_GROOVE = (214, 219, 225), (84, 93, 105), (150, 158, 168)


def draw(size):
    ss = 4
    S = size * ss
    k = S / 256.0                                       # layout below is on a 256 grid
    small = size < 32
    im = Image.new('RGBA', (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)

    def R(*v):
        return [round(c * k) for c in v]

    ow = max(ss, round((10 if small else 6) * k))       # outline width
    cy = 166                                            # wing centre line (middle of the body)
    hw = 30 if small else 27                            # half width of the wing
    # jacket: rim around the key head plus the wing, one orange part
    d.rounded_rectangle(R(4, cy - hw, 252, cy + hw), radius=round(hw * k), fill=ORANGE_DARK)
    d.rounded_rectangle(R(66, 80, 190, 250), radius=round(44 * k), fill=ORANGE_DARK)
    inset = ow / k
    d.rounded_rectangle(R(4 + inset, cy - hw + inset, 252 - inset, cy + hw - inset),
                        radius=round((hw - inset) * k), fill=ORANGE)
    d.rounded_rectangle(R(66 + inset, 80 + inset, 190 - inset, 250 - inset),
                        radius=round((44 - inset) * k), fill=ORANGE)
    if not small:                                       # push pads at the wing tips
        for x in (34, 222):
            d.ellipse(R(x - 15, cy - 15, x + 15, cy + 15), fill=ORANGE_LIGHT)
            d.ellipse(R(x - 11, cy - 11, x + 11, cy + 11), fill=ORANGE)

    # blade, from the head up to the tip (drawn over the rim, under the head)
    bw = 17 if small else 13                            # half width
    tip = [(128 - bw, 104), (128 - bw, 30), (128 - bw + 6, 12), (128, 6), (128 + bw - 6, 12), (128 + bw, 30),
           (128 + bw, 104)]
    d.polygon([(round(x * k), round(y * k)) for x, y in tip], fill=BLADE_EDGE)
    e = ow / k * 0.8
    inner = [(128 - bw + e, 104), (128 - bw + e, 31), (128 - bw + 6 + e * 0.6, 14 + e), (128, 6 + e * 1.6),
             (128 + bw - 6 - e * 0.6, 14 + e), (128 + bw - e, 31), (128 + bw - e, 104)]
    d.polygon([(round(x * k), round(y * k)) for x, y in inner], fill=BLADE)
    if not small:                                       # the cut track of a laser-cut blade
        d.rounded_rectangle(R(124, 24, 132, 92), radius=round(4 * k), fill=BLADE_GROOVE)

    # key head
    d.rounded_rectangle(R(84, 96, 172, 236), radius=round(30 * k), fill=HEAD_EDGE)
    d.rounded_rectangle(R(84 + inset, 96 + inset, 172 - inset, 236 - inset), radius=round((30 - inset) * k),
                        fill=HEAD)
    if size >= 48:                                      # the snap lips reach over the long sides of the key
        for x0, x1 in ((80, 95), (161, 176)):
            d.rounded_rectangle(R(x0, 122, x1, 210), radius=round(6 * k), fill=ORANGE_DARK)
            d.rounded_rectangle(R(x0 + inset, 122 + inset, x1 - inset, 210 - inset),
                                radius=round((6 - inset) * k), fill=ORANGE)
    if not small:
        for y in (138, 180):                            # two buttons
            d.ellipse(R(128 - 17, y - 17, 128 + 17, y + 17), fill=BUTTON_EDGE)
            d.ellipse(R(128 - 14, y - 14, 128 + 14, y + 14), fill=BUTTON)
        d.ellipse(R(128 - 9, 213 - 9, 128 + 9, 213 + 9), fill=(0, 0, 0, 0))       # ring hole
        d.ellipse(R(128 - 9, 213 - 9, 128 + 9, 213 + 9), outline=BUTTON, width=max(ss, round(3 * k)))
    else:
        d.ellipse(R(128 - 22, 150 - 22, 128 + 22, 150 + 22), fill=BUTTON)           # one button
    return im.resize((size, size), Image.LANCZOS)


def main():
    imgs = {s: draw(s) for s in SIZES}
    imgs[256].save(HERE / 'fobfit.png')
    imgs[256].save(HERE / 'fobfit.ico', sizes=[(s, s) for s in SIZES],
                   append_images=[imgs[s] for s in SIZES if s != 256])
    sheet = Image.new('RGBA', (sum(SIZES) + 10 * len(SIZES) + 10, 276), (245, 245, 245, 255))
    x = 10
    for s in SIZES:
        sheet.alpha_composite(imgs[s], (x, 10))
        x += s + 10
    return sheet


if __name__ == '__main__':
    sheet = main()
    print('wrote', HERE / 'fobfit.ico', HERE / 'fobfit.png')
