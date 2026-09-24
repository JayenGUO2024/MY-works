import json
import random

# 可以根据需要，增减或替换以下短语，以便生成更加丰富的评论
service_intros = [
    "Despite multiple attempts to seek assistance,",
    "Even after contacting support multiple times,",
    "I reached out to the service team several times, but",
    "I called customer support numerous times; however,"
]

service_problems = [
    "the customer support team remained unresponsive",
    "no representative ever got back to me",
    "they never returned my calls or emails",
    "they couldn't solve my issue in a timely manner"
]

instructions_problems = [
    "the instructions were vague at best",
    "the manual was nearly impossible to follow",
    "the setup guide lacked crucial details",
    "the provided documentation was very confusing"
]

overall_consequence = [
    "turning the setup process into an unnecessarily challenging and frustrating experience.",
    "making it incredibly hard to get the device running properly.",
    "leading to a lot of wasted time during installation.",
    "resulting in a very frustrating initial experience."
]

delivery_phrases = [
    "The delivery arrived much later than promised,",
    "My package was delayed for over a week,",
    "The shipping was incredibly fast, arriving days earlier than expected,",
    "The package got lost in transit and arrived way too late,"
]

quality_phrases_negative = [
    "the build quality seemed rather flimsy",
    "the device started malfunctioning after a few days",
    "the product materials felt cheap to the touch",
    "the overall quality was not up to expectations"
]

quality_phrases_positive = [
    "the build quality was outstanding",
    "the device performed flawlessly for weeks",
    "the materials used felt premium and sturdy",
    "the product’s overall quality exceeded my expectations"
]

price_phrases_negative = [
    "the price felt unreasonably high for what it offered",
    "it was overpriced given its mediocre performance",
    "the cost is simply not justified by the product's features"
]

price_phrases_positive = [
    "the price was surprisingly affordable",
    "it offered excellent value for the cost",
    "the cost felt very reasonable for such quality"
]

packaging_phrases_negative = [
    "the packaging was torn apart upon arrival",
    "the box arrived badly damaged",
    "the product was rattling around loosely inside"
]

packaging_phrases_positive = [
    "the packaging was secure and neatly done",
    "the box arrived in perfect condition",
    "everything was well-protected and carefully sealed"
]

# 为了与示例保持一致，我们大部分情况下都突出 "quality" 或 "service" 的负面性，
# 并在其他方面上随机赋值为空或者一些简短的负面/正面表述。
sentiments = ["negative", "neutral", "positive"]


def generate_one_example():
    """
    随机生成一条示例，结构如下：
    {
      "input": "...",
      "output": {
        "delivery":  {"text": "", "sentiment": "neutral"},
        "quality":   {"text": "", "sentiment": "neutral"},
        "price":     {"text": "", "sentiment": "neutral"},
        "packaging": {"text": "", "sentiment": "neutral"},
        "service":   {"text": "", "sentiment": "neutral"}
      }
    }
    """
    # 1) 随机决定要不要加上 delivery, quality, price, packaging, service 的句子
    use_delivery = random.choice([True, False])
    use_quality = random.choice([True, False])
    use_price = random.choice([True, False])
    use_packaging = random.choice([True, False])
    use_service = True  # 保持大多数情况下都提到 service（符合您示例中的主体）

    # 2) 逐句组装 input 字符串
    sentences = []

    # - (a) Service + instructions (最初的核心内容)
    #   必定有一条service_intros + service_problems + instructions_problems + overall_consequence
    service_sentence = " ".join([
        random.choice(service_intros),
        random.choice(service_problems), "and",
        random.choice(instructions_problems) + ",",
        random.choice(overall_consequence)
    ])
    sentences.append(service_sentence)

    # - (b) 如果使用 delivery，就拼接一条和送货相关的句子
    if use_delivery:
        sentences.append(random.choice(delivery_phrases))

    # - (c) 如果使用 quality，就拼接一条和质量相关的句子
    if use_quality:
        # 随机决定正面或负面
        if random.random() < 0.5:
            # negative
            sentences.append(random.choice(quality_phrases_negative))
        else:
            # positive
            sentences.append(random.choice(quality_phrases_positive))

    # - (d) 如果使用 price，就拼接一条和价格相关的句子
    if use_price:
        if random.random() < 0.5:
            sentences.append(random.choice(price_phrases_negative))
        else:
            sentences.append(random.choice(price_phrases_positive))

    # - (e) 如果使用 packaging，就拼接一条和包装相关的句子
    if use_packaging:
        if random.random() < 0.5:
            sentences.append(random.choice(packaging_phrases_negative))
        else:
            sentences.append(random.choice(packaging_phrases_positive))

    # 最终 input：用空格拼接所有句子
    final_input = " ".join(sentences).strip()

    # 3) 构建 output
    #    - 根据是否实际使用了某个方面，提取对应 text 并随机给 sentiment
    #    - 符合您给的“当方面未提及时，text 为空、sentiment=neutral”的模式
    output_data = {
        "delivery": {
            "text": "",
            "sentiment": "neutral"
        },
        "quality": {
            "text": "",
            "sentiment": "neutral"
        },
        "price": {
            "text": "",
            "sentiment": "neutral"
        },
        "packaging": {
            "text": "",
            "sentiment": "neutral"
        },
        "service": {
            "text": "",
            "sentiment": "neutral"
        }
    }

    # 处理 service（在示例中，service 主要是“客服未响应”的情况）
    # 我们把“service_sentence”的前半部分放进 service 的 text
    # 把“instructions_problems”放进 quality 的 text
    # 并设定 sentiment 都是 negative（与原示例相仿）
    if use_service:
        # 为了模拟示例：“service”提到客服问题；“quality”提到说明书的问题
        # 先把整句拆分一下，或者只取前面一部分作为 service
        # 其实只要把前半句(客服相关)放到 service 的 text，后半句(说明书)放到 quality 的 text，即可
        # service_sentence = "... the customer support team remained unresponsive and the instructions were vague ..., turning..."
        # 我们可以简单处理一下:
        #   1) 找到 ' and ' 作为分割
        #   2) 前半给 service，后半给 quality
        # 如果没有找到，就默认全部给 service
        if "and" in service_sentence:
            parts = service_sentence.split("and", 1)  # 只拆分一次
            service_text_part = parts[0].strip()
            rest_part = parts[1].strip()
        else:
            service_text_part = service_sentence
            rest_part = ""

        # 再看后半是否包含说明书或 setup 之类
        # 如果包含了 instructions / setup 等关键词，就给 quality
        if any(k in rest_part.lower() for k in ["instruction", "manual", "setup", "process"]):
            # 让 quality的 text = rest_part
            output_data["quality"]["text"] = rest_part
            output_data["quality"]["sentiment"] = "negative"
        else:
            # 否则也都放到 service
            service_text_part += " " + rest_part

        output_data["service"]["text"] = service_text_part
        output_data["service"]["sentiment"] = "negative"

    # 如果使用了 quality，并且上面并未写进去 text（说明书）的话，就把第三步中随机生成的那句 quality phrase 放进去
    if use_quality and output_data["quality"]["text"] == "":
        # 这时说明 quality 可能是正面的或负面的句子
        # 重新生成一下随机的 phrase
        possible_phrases = quality_phrases_negative + quality_phrases_positive
        chosen_phrase = random.choice(possible_phrases)
        output_data["quality"]["text"] = chosen_phrase
        # 给 sentiment
        if chosen_phrase in quality_phrases_negative:
            output_data["quality"]["sentiment"] = "negative"
        else:
            output_data["quality"]["sentiment"] = "positive"

    # 如果使用了 price
    if use_price:
        # 随机决定 negative / positive
        # 并写到 output 的 text
        pneg = random.choice(price_phrases_negative)
        ppos = random.choice(price_phrases_positive)
        # 我们让 50% 概率选负面
        if random.random() < 0.5:
            output_data["price"]["text"] = pneg
            output_data["price"]["sentiment"] = "negative"
        else:
            output_data["price"]["text"] = ppos
            output_data["price"]["sentiment"] = "positive"

    # 如果使用了 delivery
    if use_delivery:
        # 在 delivery_phrases 中随机选一句
        # 如果包含 late/delay/lost 则是 negative, 否则 positive
        chosen_deliv = random.choice(delivery_phrases)
        output_data["delivery"]["text"] = chosen_deliv
        if any(k in chosen_deliv.lower() for k in ["late", "lost", "delayed"]):
            output_data["delivery"]["sentiment"] = "negative"
        elif any(k in chosen_deliv.lower() for k in ["fast", "earlier"]):
            output_data["delivery"]["sentiment"] = "positive"
        else:
            output_data["delivery"]["sentiment"] = "neutral"

    # 如果使用了 packaging
    if use_packaging:
        # 在 packaging_phrases 中随机选一句
        chosen_pack = random.choice(packaging_phrases_negative + packaging_phrases_positive)
        output_data["packaging"]["text"] = chosen_pack
        if chosen_pack in packaging_phrases_negative:
            output_data["packaging"]["sentiment"] = "negative"
        else:
            output_data["packaging"]["sentiment"] = "positive"

    # 构建最终记录
    record = {
        "input": final_input,
        "output": output_data
    }

    return record


def main():
    # 准备生成约 1000 条
    n = 1000
    results = []

    for _ in range(n):
        record = generate_one_example()
        results.append(record)

    # 将结果输出到一个 JSON 文件（或您可以改成 CSV、打印等）
    with open("samples.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"已生成 {n} 条示例数据，并保存在 samples.json 文件中。")


if __name__ == "__main__":
    main()
