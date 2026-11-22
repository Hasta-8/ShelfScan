# Grouping labeller (experiment S4)

A local tool for labelling which products on a shelf are the same, so the grouping
step can be scored. Labels are saved to `eval/labels/` (git-ignored: the images are
SKU-110K's, licensed for non-commercial use without redistribution).

```bash
uv run --group eval python eval/labeler/server.py \
  "/media/debadri/sda5/Git Repos/Datasets/SKU110K/data/test-00000-of-00007.parquet"
# then open http://127.0.0.1:8002
```

24 images, about 110 boxes each. Work saves per image; stop and resume any time
(the menu marks finished images ✓ and opens the next unfinished one).

## The rule

- **Same product** = same brand, same variant/flavour, same packaging. If two items
  look identical but one is visibly bigger, they are different products.
- **Brand** = the maker's brand on the pack (e.g. "Pond's", "Starbucks"). Type
  `unknown` if you can't read it; reuse the exact same spelling across images (the
  box suggests names you've used before).
- **Unsure** = you can't tell what the item is (cut off, hidden, blurred, too small).
  Unsure boxes are left out of scoring, so use it instead of guessing.

## Fast workflow

1. **Zoom in** with the mouse wheel; **right-drag** to move; **0** resets.
2. **Drag a rectangle** over a run of identical products (a box is selected when its
   centre is inside). **Shift-drag** or **shift-click** adds more boxes.
3. Press **N**: they become a new product, and the cursor jumps to its brand field.
   Type the brand, press Enter.
4. Found more of the same product elsewhere? Select them and press **A** (adds to the
   active product, the one highlighted on the right; click a product there to make
   it active).
5. Items that each appear once: select them and press **S** (each becomes its own
   product), then select several of the same brand and press **B** to name the brand
   once for all of them.
6. **U** marks unsure, **X** clears a label, **Z** undoes, **Esc** clears the selection.
7. **Save & next** unlocks when every box is a product or unsure and every product has
   a brand. **Remaining → own products** turns all unlabelled boxes into one-item
   products (then brand them with B).

Hovering a box shows it enlarged at the top right, with its current label.
