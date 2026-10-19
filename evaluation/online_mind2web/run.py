import argparse
import os
from methods.agenttrek_eval import *
from methods.automomous_eval import *
from methods.webjudge_general_eval import *
from methods.webjudge_online_mind2web import *
from methods.webvoyager_eval import *
from utils import OpenaiEngine, extract_predication
import json
import copy
import asyncio
import multiprocessing
import math

def auto_eval(args, task_subset, final_predicted_labels, lock, model):

    ################## get the already done task id ###############
    output_json_path = os.path.join(args.output_path, f"{args.mode}_{args.model}_score_threshold_{args.score_threshold}_auto_eval_results.json")
    already_ids = []
    if os.path.exists(output_json_path):
        with open(output_json_path,"r") as f:
            already_data = f.read()
        already_tasks = already_data.splitlines()
        for item in already_tasks:
            item = json.loads(item)
            already_ids.append(item["task_id"])

    print(f"The number of already done tasks: {len(already_ids)}")

    for task_id in task_subset:
        #Skip already done task
        if task_id in already_ids:
            continue

        trajectory_images_path = os.path.join(args.trajectories_dir, task_id, "trajectory")
        screenshot_paths = []
        thoughts = None
        action_history = None
        final_result_response = None
        input_image_paths = None
        task_description = None
        # Load results
        with open(os.path.join(args.trajectories_dir, task_id, "result.json")) as f:
            result = json.load(f)
            output_results = copy.deepcopy(result)
            task_description = result["task"]

            is_v2 = result.get("schema_version") == "online-mind2web-v2"

            if is_v2:
                # v2: action_history is a list of ActionStep objects
                steps = result["action_history"]
                action_history = [step["action"] for step in steps]
                thoughts = [step.get("thought") for step in steps]
                # Build screenshot paths from each step's screenshot field
                screenshot_paths = [
                    os.path.join(trajectory_images_path, step["screenshot"])
                    for step in steps
                ]
                final_result_response = result.get("agent_final_answer")
                input_image_paths = result.get("input_image_paths")
            else:
                # v1: flat string arrays
                if "action_history" in result:
                    action_history = result["action_history"]
                if "thoughts" in result:
                    thoughts = result["thoughts"]
                if "final_result_response" in result:
                    final_result_response = result["final_result_response"]
                if "input_image_paths" in result:
                    input_image_paths = result["input_image_paths"]

        text_observations = result.get("text_observations")

        print(f"Start evaluation for {task_description}")

        # For v1, build screenshot_paths from directory listing
        if not is_v2:
            for image in sorted(os.listdir(trajectory_images_path), key=lambda x: int(re.findall(r'\d+', x)[0])):
                screenshot_paths.append(os.path.join(trajectory_images_path, image))

        # Do the auto-eval
        if args.mode == "Autonomous_eval":
            messages, text, system_msg = Autonomous_eval(task_description, action_history, screenshot_paths[-1])

        elif args.mode == "AgentTrek_eval":
            messages, text, system_msg = AgentTrek_eval(task_description, action_history, thoughts, screenshot_paths[-1])

        elif args.mode == "WebVoyager_eval":
            messages, text, system_msg = WebVoyager_eval(task_description, screenshot_paths, final_result_response)

        elif args.mode == "WebJudge_Online_Mind2Web_eval":
            messages, text, system_msg, record, key_points = asyncio.run(WebJudge_Online_Mind2Web_eval(task_description, action_history, screenshot_paths, model, args.score_threshold, final_answer=final_result_response, text_observations=text_observations))
            output_results["image_judge_record"] = record
            output_results["key_points"] = key_points

        elif args.mode == "WebJudge_general_eval":
            messages, text, system_msg, record, key_points = asyncio.run(WebJudge_general_eval(task_description, input_image_paths, thoughts, action_history, screenshot_paths, model, args.score_threshold))
            output_results["image_judge_record"] = record
            output_results["key_points"] = key_points

        else:
            raise ValueError(f"Unknown mode: {args.mode}")

        # max_new_tokens 默认 512 太小:WebJudge 要逐条核对 key points 再给 Status,
        # 实测中位 1969 字符、最长 2685,512 token 会在写到 Status 之前被截断。
        # 而 extract_predication 找不到 'status:' 就 except -> return 0,**静默记 0 分**。
        # 实测受损:base 29.3% / op-0821 23.3% / hybrid 39.7% 的判词没有 status,全被记 0。
        # 2048 仍有约 6% 撞顶(判官会反复推翻自己重新分析,写到 8000~10000 字符),故用 4096。
        # 轨迹越长越容易中招,所以这个偏差对步数多的模型不利,不是随机噪声。
        response = model.generate(messages, max_new_tokens=4096)[0]
        predicted_label = extract_predication(response, args.mode)
        # 反爬阻断:独立于 success/failure 的第三个轴,判词里多出的一行 `Blocked: yes/no`。
        # 解析不到时记 None(不是 False)—— 把"判官没写"和"判官说没被挡"混成一个数,
        # 阻断率会被系统性低估。判据与另两套判官共用 evaluation/hywebench/blocked_criterion.py。
        _blocked = None
        try:
            import sys as _s, os as _o
            _s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)),
                                           "..", "hywebench"))
            from blocked_criterion import parse_blocked as _pb
            _blocked = _pb(response)
        except Exception:
            pass

        #Store evaluation details
        evaluation_results = {"response": response, "predicted_label": predicted_label,
                              "blocked": _blocked}
        output_results["task_id"] = task_id
        output_results["input_text"] = text
        output_results["system_msg"] = system_msg
        output_results["evaluation_details"] = evaluation_results
        output_results["predicted_label"] = predicted_label
        output_results["blocked"] = _blocked

        with lock:
            final_predicted_labels.append(predicted_label)

        print(f"Finish evaluation for {task_description}")
        print("="*20)
        os.makedirs(args.output_path, exist_ok=True)
        with lock:
            with open(os.path.join(args.output_path, f"{args.mode}_{args.model}_score_threshold_{args.score_threshold}_auto_eval_results.json"), "a+") as f_out:
                f_out.write(json.dumps(output_results) + "\n")


def process_subset(task_subset, args, final_predicted_labels, lock, model):

    auto_eval(args, task_subset, final_predicted_labels, lock, model)

def parallel_eval(args, num_workers=60):

    #Evaluate in parallel based on num of works
    task_dirs = [
        d for d in sorted(os.listdir(args.trajectories_dir)) 
        if os.path.isdir(os.path.join(args.trajectories_dir, d))
    ]
    print(f"Evaluating {len(task_dirs)} tasks in total.")
    if not task_dirs:
        print("No trajectories to judge.")
        return
    num_workers = max(1, min(num_workers, len(task_dirs)))
    chunk_size = math.ceil(len(task_dirs) / num_workers)
    task_subsets = [task_dirs[i:i + chunk_size] for i in range(0, len(task_dirs), chunk_size)]

    #Load model
    model = OpenaiEngine(
        model=args.model,
        api_key=args.api_key
    )

    lock = multiprocessing.Lock()
    with multiprocessing.Manager() as manager:
        final_predicted_labels = manager.list()
        processes = []
        for subset in task_subsets:
            p = multiprocessing.Process(target=process_subset, args=(subset, args, final_predicted_labels, lock, model))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
        if any(p.exitcode != 0 for p in processes):
            raise RuntimeError("A judge worker failed; inspect the traceback and resume the run.")

        success_num = sum(final_predicted_labels) 

    print("Evaluation complete.")
    print(f"The success rate is {(success_num / len(task_dirs)) * 100}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto evaluation of web navigation tasks.")
    parser.add_argument('--mode', type=str, default='Online_Mind2Web_eval', help='the mode of evaluation')
    parser.add_argument('--model', type=str, default='gpt-4o')
    parser.add_argument("--trajectories_dir", type=str, required=True, help="Path to trajectories directory")
    parser.add_argument("--api_key", type=str, required=True, help="The api key")
    parser.add_argument("--output_path", type=str, required=True, help="The output path")
    parser.add_argument('--score_threshold', type=int, default=3)
    parser.add_argument('--num_worker', type=int, default=60)
    args = parser.parse_args()

    parallel_eval(args, args.num_worker)
