import functools
import os
import json
import pickle
import argparse
import inspect
import importlib
import time

import torch
import numpy as np
import requests
import subprocess
from typing import get_type_hints
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from datasets.fingerprint import Hasher

from Bio import UniProt
from Bio import SeqIO, Entrez
from Bio.PDB.MMCIFParser import MMCIFParser

Entrez.email = "xxx@gmail.com"

def load_class(module_path: str, class_name: str):
    try:
        module = importlib.import_module(module_path)
        if hasattr(module, class_name):
            return getattr(module, class_name)
        else:
            raise AttributeError(f"模块 {module_path} 中不存在类 {class_name}")
    except ImportError:
        raise ImportError(f"无法导入模块 {module_path}")


def auto_argparser(func, parser, description=None):
    sig = inspect.signature(func)
    params = list(sig.parameters.values())
    params = params[1:] if params and params[0].name == 'self' else params
    if parser is None:
        parser = argparse.ArgumentParser(description=description or f"Auto-generated parser")
    type_hints = get_type_hints(func)
    for param in params:
        param_name = param.name
        param_type = type_hints.get(param_name, None)
        if param.default is not inspect.Parameter.empty:
            args = [f"--{param_name}"]
            kwargs = {
                "default": param.default,
                "help": f"Default: {param.default}"
            }
            if param_type is None and param.default is not None:
                param_type = type(param.default)
        else:
            args = [f"--{param_name}"]  # 推荐用--参数，更明确
            kwargs = {"required": True, "help": "Required parameter"}
        if param_type is bool:
            if param.default is False:
                kwargs["action"] = "store_true"
                del kwargs["default"]
            elif param.default is True:
                kwargs["action"] = "store_false"
                del kwargs["default"]
            # del kwargs["type"]
        elif param_type in (list, tuple):
            kwargs["nargs"] = "+"
            kwargs["type"] = str
        elif param_type is not None:
            kwargs["type"] = param_type
        parser.add_argument(*args, **kwargs)
    return parser



def get_func_arguments(func, *args, **kwargs):
    sig = inspect.signature(func)
    params = list(sig.parameters.values())
    if params and params[0].name == 'self':
        params = params[1:]
        new_sig = sig.replace(parameters=params)
    else:
        new_sig = sig
    bound_args = new_sig.bind(*args, **kwargs)
    bound_args.apply_defaults()  # 填充默认值
    arguments = {}
    for param in params:
        param_name = param.name
        arguments[param_name] = bound_args.arguments.get(param_name)
    return arguments


def get_uniprot_info(refseq_id=None, gene_name=None, organism_id=9606, save_dir=None, return_all=False, reviewed=False):
    assert refseq_id or gene_name
    save_file = os.path.join(save_dir, f'{refseq_id if refseq_id else gene_name}.json')
    if save_dir is not None and os.path.exists(save_file):
        with open(save_file) as f:
            return json.load(f)
    query = (
        f"(xref:refseq-{refseq_id}) " if refseq_id is not None else  f"(gene:{gene_name}) ",
        f"AND (organism_id:{organism_id}) ", #9606 homo sapiens
        f"AND (reviewed:'true' )" if reviewed else ""  # 仅保留人工审核的高质量条目
    )
    query = " ".join(query)
    results = UniProt.search(
        query,
        fields=["accession", "id", "sequence", "gene_names", "protein_name"] if not return_all else None,
    )
    time.sleep(0.5)
    uniprot_data = []
    for entry in results:
        uniprot_data.append(entry)
        # if entry.get("sequence") is None:
        #     continue
        # sequence = entry.get("sequence", {}).get("value")
        # uniprot_data.append({
        #     "uniprot_id": entry["primaryAccession"],
        #     "entry_name": entry["uniProtkbId"],
        #     "gene_name": entry["genes"][0]["geneName"]["value"],
        #     "protein_name": entry["proteinDescription"]["recommendedName"]["fullName"]["value"],
        #     "sequence": sequence
        # })
    if len(uniprot_data) == 0:
        return None
    if save_dir is not None:
        with open(os.path.join(save_dir, f"{refseq_id}.json"), "w") as f:
            json.dump(uniprot_data, f, indent=2)
    return uniprot_data


def get_ncbi_info(accession, rettype="gb", save_dir=None):
    save_file = os.path.join(save_dir, f"{accession}.{rettype}")
    if save_dir is not None and os.path.exists(save_file):
        return SeqIO.read(save_file, rettype)
    try:
        # 步骤1: 使用esearch查找对应的ID
        with Entrez.esearch(db="protein", term=accession, retmax=1) as handle:
            search_results = Entrez.read(handle)
            id_list = search_results["IdList"]
            if not id_list:
                raise ValueError(f"未找到登录号 {accession} 的记录")
        # 步骤2: 使用efetch获取具体数据
        with Entrez.efetch(
                db="protein",
                id=id_list[0],
                rettype=rettype,
                retmode="text"
        ) as handle:
            # 解析结果（GenBank格式包含注释，FASTA仅含序列）
            if rettype == "gb":
                record = SeqIO.read(handle, "genbank")
            elif rettype == "fasta":
                record = SeqIO.read(handle, "fasta")
            else:
                raise ValueError(f"不支持的返回格式: {rettype}")
        time.sleep(0.4)
        if save_dir is not None:
            SeqIO.write(record, save_file, rettype)
        return record
    except Exception as e:
        print(f"获取数据失败: {str(e)}")
        return None


class MultiThreadProcessor:
    def __init__(self, max_workers: int = 5):
        self.max_workers = max_workers
        self.results = []
        self.failed_tasks = []

    def process_task(self, task, task_func):
        try:
            result = task_func(task)
            return {
                "task": task,
                "result": result,
                "status": "success"
            }
        except Exception as e:
            print(f"任务处理失败: {task}, 错误: {str(e)}")
            return {
                "task": task,
                "result": None,
                "status": "failed",
                "error": str(e)
            }

    def process_batch(self, tasks, task_func):
        self.results = []
        self.failed_tasks = []
        if tasks is None or len(tasks) == 0:
            print("任务列表为空，无需处理")
            return
        with tqdm(total=len(tasks), desc="处理进度", unit="任务") as pbar:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(self.process_task, task, task_func): task
                    for task in tasks
                }
                for future in as_completed(futures):
                    result = future.result()
                    if result["status"] == "failed" or result['result'] is None:
                        self.failed_tasks.append(result)
                    else:
                        self.results.append(result)
                    pbar.update(1)
        success_count = len(self.results)
        print(f"任务处理完成 - 成功: {success_count}, 失败: {len(self.failed_tasks)}, 总任务: {len(tasks)}")

    def get_success_results(self):
        return [r for r in self.results]

    def get_failed_tasks(self):
        return self.failed_tasks


def read_mmcif_structure(mmcif_file):
    if not os.path.exists(mmcif_file):
        raise FileNotFoundError(f"file not found: {mmcif_file}")
    parser = MMCIFParser(QUIET=True)  # QUIET=True关闭警告信息
    structure_id = os.path.splitext(os.path.basename(mmcif_file))[0]
    with open(mmcif_file, 'r') as f:
        structure = parser.get_structure(structure_id, f)  # 传入文件对象而非路径
    return structure


def exec_command(command, verbose=False, shell=True, skip_error=False):
    if verbose:
        print(command)
    p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=os.environ, shell=shell)
    returncode = p.poll()
    if verbose:
        while returncode is None:
            out = p.stdout.readline().strip().decode()
            if len(out)!=0:
                print(out)
            returncode = p.poll()
        out = p.stdout.readline().strip().decode()
        if len(out) != 0:
            print(out)
    stdout, stderr = p.communicate()
    if p.returncode and not skip_error:
        raise Exception('failed: on %s\n%s\n%s\n%s' % (command, stderr, stdout.decode(), p.returncode))
    if stdout is not None:
        return stdout.decode().strip()


def exec_commands(commands, batch_size=None, desc="exec"):
    batch_size = os.cpu_count()//2 if batch_size is None else batch_size
    for i in tqdm(range(0, len(commands), batch_size), desc=desc):
        j = min(len(commands), i+batch_size)
        ps = [(subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=os.environ, shell=True), command) for command in commands[i:j]]
        for p, cmd in ps:
            stdout, stderr = p.communicate()
            if p.returncode:
                raise Exception('failed: on %s\n%s\n%s' % (cmd, stderr, stdout.decode()))


class Cache():
    def __init__(self, func, disk_cache_dir, overwrite=False):
        functools.update_wrapper(self, func)
        self.func = func
        self.disk_cache_dir = disk_cache_dir
        self.overwrite = overwrite

    def __call__(self, *args, **kwargs):
        disk_cache_dir = self.disk_cache_dir
        if len(args)!=0 and hasattr(args, '__class__'):
            params = tuple([*args[1:], *kwargs.items()])
            disk_cache_dir = getattr(args[0], 'disk_cache_dir', disk_cache_dir)
        else:
            params = tuple([*args, *kwargs.items()])
        hash = Hasher.hash(params)
        save_file = os.path.join(disk_cache_dir, f"{hash}.pkl")
        if self.overwrite or not os.path.exists(save_file):
            ans = self.func(*args, **kwargs)
            with open(save_file, 'wb') as f:
                pickle.dump(ans, f)
        with open(save_file, 'rb') as f:
            ans = pickle.load(f)
        return ans

    def __get__(self, instance, owner):
        return functools.partial(self.__call__, instance)



def disk_cache(func=None, disk_cache_dir=None, overwrite=False):
    if func is None:
        return lambda x: Cache(x, disk_cache_dir=disk_cache_dir, overwrite=overwrite)
    else:
        return Cache(func, disk_cache_dir=disk_cache_dir, overwrite=overwrite)

def serialize3d(pos, batch_idx=None, order='hilbert', grid_size=0.02):
    from repo.PointMamba.models.serialization import Point
    if isinstance(order, str):
        order = [order]
    if isinstance(pos, np.ndarray):
        pos = torch.from_numpy(pos)
    if isinstance(batch_idx, np.ndarray):
        batch_idx = torch.from_numpy(batch_idx)
    if batch_idx is None:
        batch_idx = torch.zeros(len(pos), dtype=torch.int)
    scaled_coord = pos / grid_size
    grid_coord = torch.floor(scaled_coord).to(torch.int64)
    min_coord = grid_coord.min(dim=1, keepdim=True)[0]
    grid_coord = grid_coord - min_coord
    point_dict = {'batch': batch_idx, 'grid_coord': grid_coord, }
    point_dict = Point(**point_dict)
    point_dict.serialization(order=order)
    order = point_dict.serialized_order.squeeze()
    inverse_order = point_dict.serialized_inverse.squeeze()
    return order.numpy(), inverse_order.numpy()
