# 단순 원본 이미지
prompt_img_org = """
Describe this image in a strict JSON format:
{
"sex": man / woman / male baby / female baby
"object": "nothing" if no object belonging to the person, "{object in body parts}" if the person has an object e.g. "a pencil on the person's right hand",
"outfit": an outfit type
"pose": a description of the person's pose, regardless of belongings to the person, in a phrase,
"background": a short description of the person's surrounding backgrounds, in a phrase
}
"""

# qwen_image_edit 결과 이미지
prompt_nano_org = """
Describe this image in a strict JSON format:
{
"sex": man / woman / male baby / female baby,
"pose": a description of the person's pose, regardless of belongings to the person, in a phrase,
"background": a short description of the person's surrounding backgrounds, in a phrase
}
"""

# 원본에 키포인트 렌더링된 이미지
prompt_nano_rendered = """
Describe this rendered image in a strict JSON format:
{
"sex": man / woman / male baby / female baby
"pose": a description of the person's pose, regardless of belongings to the person, in a phrase
"background": a short description of the person's surrounding backgrounds, in a phrase
}
"""