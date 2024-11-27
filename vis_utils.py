from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
import matplotlib.pyplot as plt


def plot_inset(
    img,
    ssim,
    psnr,
    zoom=1.25,
    bbox_anchor=(1.15, 0.235),
    x_min=130,
    x_max=155,
    y_min=100,
    y_max=125,
    stats=True,
    phase_img=False,
):
    extent = (0, img.shape[1], 0, img.shape[0])

    fig, axe = plt.subplots(nrows=1, ncols=1)
    if phase_img:
        axe.imshow(img, extent=extent)
        axe.axis("off")
        alt_color = "tab:orange"
        if stats:
            axe.text(
                3,
                img.shape[0] - 13,
                f"APD {ssim:.3f}",
                c=alt_color,
                # backgroundcolor="gray",
                fontsize="14",
                bbox={"alpha": 0.7, "linewidth": 0, "facecolor": "k"},
            )
    else:
        alt_color = "y"
        axe.imshow(img, extent=extent, cmap="gray")
        axe.axis("off")
        if stats:
            axe.text(
                3, img.shape[0] - 13, f"SSIM: {ssim:.2f}", c=alt_color, fontsize="14"
            )
            axe.text(
                3, img.shape[0] - 25, f"PSNR: {psnr:.1f}", c=alt_color, fontsize="14"
            )

    zoom_axe = zoomed_inset_axes(
        axe,
        zoom=zoom,
        bbox_to_anchor=bbox_anchor,
        bbox_transform=axe.transAxes,
        borderpad=0,
    )
    if phase_img:
        zoom_axe.imshow(img, extent=extent)
    else:
        zoom_axe.imshow(img, extent=extent, cmap="gray")
    zoom_axe.set_xlim(x_min, x_max)
    zoom_axe.set_ylim(y_min, y_max)

    zoom_axe.tick_params(
        top=False,
        bottom=False,
        left=False,
        right=False,
        labelleft=False,
        labelbottom=False,
    )
    for axis in ["top", "bottom", "left", "right"]:
        zoom_axe.spines[axis].set_color(alt_color)
        zoom_axe.spines[axis].set_linewidth(3)

    mark_inset(axe, zoom_axe, loc1=1, loc2=3, fc="none", ec=alt_color)
    return fig
