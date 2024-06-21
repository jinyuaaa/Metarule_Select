import config
import asyncio

def case_writer(label):
    max_pic_num = config.config['max_pic_num']
    label_pl_map = config.config['label_pl_map']
    case_str = 'f(['
    A_str_list = []
    B_str_list = []
    # the last label "frame type" is not writen in the pl file
    for i in range(len(label)-1):
        if i < max_pic_num:
            if label_pl_map[label[i]] is not None:
                A_str_list.append(label_pl_map[label[i]])
        elif i == max_pic_num:
            B_str_list.append(label_pl_map[label[i]])
        else:
            B_str_list.append(str(label[i]))
    A_str = ','.join(A_str_list)
    B_str = ','.join(B_str_list)
    case_str = case_str + A_str + '],[' + B_str + '])'
    return case_str


def task_writer(task_name, pos_label, neg_label):
    file_name = config.config['pl_path']+ 'task_' + task_name + '.pl'
    with open(file_name, 'w') as f:
        f.write(":- ['task" + "_" + task_name + "_bk.pl'].\n")
        f.write("\n")

        pos_list = []
        pos_str = "a :- Pos=["
        for j in range(len(pos_label)):
            pos_list.append(case_writer(pos_label[j]))
        pos_str = pos_str + ',\n          '.join(pos_list) + '],\n'

        neg_list = []
        neg_str = "     Neg=["
        for j in range(len(neg_label)):
            neg_list.append(case_writer(neg_label[j]))
        neg_str = neg_str + ',\n          '.join(neg_list) + '],\n'
        f.write(pos_str+neg_str+'     learn(Pos,Neg).')

def bk_writer(task_name, metarule_statu):
    file_name = config.config['pl_path'] + 'task_' + task_name + '_bk.pl'
    num_metarule = config.config['num_metarule']
    metarule_pl_map = config.config['metarule_pl_map']

    with open(file_name, 'w') as f:
        f.write(config.bk_prepare[0])
        for i in range(num_metarule):
            if metarule_statu[i] == 1.0:
                f.write(metarule_pl_map[i])
        f.write("\n")
        for i in config.bk_prepare[1:]:
            f.write(i)

async def run_pl(file_path):
    cmd = "/usr/bin/swipl --stack-limit=8g -s {} -g a -t halt".format(file_path)
    proc = await asyncio.create_subprocess_shell(
        cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE)

    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=config.config['pl_time_limit'])
        if proc.returncode == 0:
            return 0, stdout.decode('UTF-8')
        else:
            return -1, stderr.decode('UTF-8')
    except asyncio.TimeoutError as e:
        if proc.returncode is None:
            try:
                proc.kill()
            except OSError:
                # Ignore 'no such process' error
                pass
        return -2, "Timeout " + str(e)  # timeout error


def read_pl_out(pl_out_str):
    prog_list = []
    for line in pl_out_str.splitlines():
        if line[0] == 'f':
             prog_list.append(line)
    prog_str = '\n'.join(prog_list)
    return prog_str


