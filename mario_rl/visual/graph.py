import cv2


def draw_text(img, *, text, x, y, font_face=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.5, font_thickness=1,
              font_color=(255, 255, 255), font_alpha=1.0):
    """Alpha blend text onto image."""
    overlay = img.copy()
    cv2.putText(overlay, text, (x, y), font_face, font_scale, font_color, font_thickness)
    return cv2.addWeighted(overlay, font_alpha, img, 1 - font_alpha, 0)


def draw_rectangle(img, *, x, y, width, height, color=(255, 255, 255), alpha=1.0):
    """Alpha blend rectangle onto image."""
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + width, y + height), color, -1)
    return cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)


def draw_horizontal_bar_graph(img, *, values, labels, colors, x, y, width, label_width, margin=5, alpha=0.8,
                              alphas=(1.0,),
                              max_value=1.0, value_format="{:.2f}",
                              font_size=0.5, font_face=cv2.FONT_HERSHEY_SIMPLEX, font_thickness=1,
                              font_colors=((255, 255, 255),),
                              background_color=(0, 0, 0)):
    """Draw a horizontal bar graph on the image.

    :param img: image to draw
    :param values: values to draw
    :param labels: labels to draw
    :param colors: colors to draw
    :param x: x position to draw
    :param y: y position to draw
    :param width: width of graph
    :param label_width: width of label area
    :param margin: margin of bar graph
    :param alpha: alpha of graph area
    :param alphas: alpha of each bar graph (if length of this is 1, use this alpha for all bar graphs)
    :param max_value: maximum value of value to draw
    :param value_format: format of value to draw
    :param font_size: font size of label
    :param font_face: font face of label
    :param font_thickness: font thickness of label
    :param font_colors: font color of each label (if length of this is 1, use this color for all labels)
    :param background_color: background color of label and bar
    :return: image with bar graph
    """
    text_width, text_height = cv2.getTextSize("A", font_face, font_size, font_thickness)[0]
    each_height = text_height + margin
    label_x = x + margin
    label_y = y + text_height + margin
    bar_x = label_x + label_width
    bar_y = y + margin
    bar_width = width - label_width - margin * 2
    bar_height = text_height
    # draw a background
    img = draw_rectangle(img, x=x, y=y, width=width, height=each_height * len(values) + margin,
                         color=background_color, alpha=alpha)
    if len(font_colors) == 1:
        font_colors = font_colors * len(values)
    if len(alphas) == 1:
        alphas = alphas * len(values)
    for value, label, color, font_color, alpha in zip(values, labels, colors, font_colors, alphas):
        # draw a label on the left
        img = draw_text(img, text=label, x=label_x, y=label_y,
                        font_face=font_face, font_scale=font_size, font_color=font_color, font_alpha=alpha, font_thickness=font_thickness)
        # draw a bar
        bar_length = int(bar_width * value / max_value)
        bar_length = min(bar_length, bar_width)
        img = draw_rectangle(img, x=bar_x, y=bar_y, width=bar_length, height=bar_height, color=color, alpha=alpha)
        # draw a value on the bar
        value_text = value_format.format(value)
        img = draw_text(img, text=value_text, x=bar_x, y=bar_y + text_height,
                        font_face=font_face, font_scale=font_size, font_color=font_color, font_alpha=alpha, font_thickness=font_thickness)
        label_y += each_height
        bar_y += each_height
    return img
