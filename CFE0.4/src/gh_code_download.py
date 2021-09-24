import os
import json
import random
import requests
from tqdm import tqdm
import scrapy
import requests
import os
import wget
import random
import subprocess

# topics
topics = ['3d', 'algorithm', 'android', 'ansible', 'api', 'aws',
'azure', 'bash', 'bitcoin', 'bot', 'cli', 'compiler', 'covid-19',
'css', 'data-structures', 'data-visualization', 'database',
'deep-learning', 'django', 'docker', 'documentation',
'ethereum', 'flask', 'framework', 'git', 'google',
'graphql', 'hacktoberfest', 'html', 'http', 'ios', 'javascript',
'jquery', 'json', 'jupyter-notebook', 'kubernetes', 'latex',
'library', 'linux', 'machine-learning', 'macos', 'markdown',
'minecraft', 'mongodb', 'monitoring', 'mysql', 'nlp', 'nodejs',
'parsing', 'php', 'postgresql', 'qt', 'raspberry-pi', 'react',
'rest-api', 'scikit-learn', 'security', 'server', 'serverless',
'shell', 'sql', 'telegram', 'tensorflow', 'terminal', 'terraform',
'testing', 'twitter', 'ubuntu', 'vim', 'webapp', 'windows', 'xml']

# tokens, the more the better
tokens = ['ef305c43723576a6ce4cdac02879a39af44d5293',
          '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',
          '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',
          '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',
          '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']
def get_token():
    return tokens[random.randint(0, len(tokens) - 1)]

# dir where save the data
repo_saving_dir = './github-repos/'

fail_list = []


# delete non-python files in dir 'path'
def clear_dir(path):
    if (os.path.islink(path)):
        os.unlink(path)
        return 0
    elif (os.path.isfile(path)):
        if (path[-3:] == '.py'):
            return 1
        else:
            os.remove(path)
            return 0
    else:
        assert os.path.isdir(path), 'This is not file, dir or link: ' + path
        cnt = 0
        for x in os.listdir(path):
            cnt = cnt + clear_dir(path + '/' + x)
        if (cnt == 0):
            os.rmdir(path)
        return cnt


def crawl_repo(item):
    # check if downloaded it before
    if (item['full_name'] in topic_map):
        return 1

    try:
        username = item['owner']['login']
        reponame = item['name']
        dirpath = repo_saving_dir + 'files/' + username + '/'
        filepath = dirpath + reponame + '.zip'
        os.system('mkdir -p ' + dirpath)

        # download and unzip
        assert os.system('curl -H "Authorization: ' + get_token()
                         + '" -L ' + item['url'] + '/zipball'
                         + ' --output ' + filepath + ' >/dev/null 2>&1') == 0, \
            'Error when downloading repo: ' + item['full_name']
        assert os.path.isfile(filepath), 'Zip pack download fail, repo name: ' + item['full_name']
        assert os.system('unzip ' + filepath + ' -d ' + dirpath + ' >/dev/null 2>&1') == 0, \
            'Error when unziping, pack name: ' + filepath

        # rename the dir name
        prefix = username + '-' + reponame
        cnt = 0
        for x in os.listdir(dirpath):
            if (os.path.isdir(dirpath + x) and x[: len(prefix)] == prefix):
                cnt += 1
                wrong_name = x
        assert cnt == 1, 'Zip pack format error, pack name: ' + filepath
        assert os.system('mv ' + dirpath + wrong_name + ' ' + dirpath + reponame) == 0, \
            'Zip pack format error, pack name: ' + filepath

        # delete non-python files
        cnt = clear_dir(dirpath + reponame)
        assert cnt <= 2000, 'Repo has to many python files, repo name: ' + item['full_name']
    except BaseException as e:
        print('[Exception]', e, flush=True)
        fail_list.append(item)
        return 0
    else:
        topic_map[item['full_name']] = item['topics']
        return 1


def crawl_topic(topic):
    # ask number of repos with given topic
    url = 'https://api.github.com/search/repositories?q=language:python+topic:' + topic + '&per_page=100&page=1'
    data = requests.get(url, headers={'Authorization': 'token ' + get_token(),
                                      'Accept': 'application/vnd.github.mercy-preview+json'}).json()
    print('-----Working on topic {' + topic + '}: try to get', min(1000, data['total_count']), 'repos-----', flush=True)

    cnt = 0
    for i in range(min(10, (data['total_count'] + 99) // 100)):
        # get the i-th page
        url = 'https://api.github.com/search/repositories?q=language:python+topic:' \
              + topic + '&per_page=100&page=' + str(i + 1)
        try:
            data = requests.get(url, headers={'Authorization': 'token ' + get_token(),
                                              'Accept': 'application/vnd.github.mercy-preview+json'}).json()
        except BaseException as e:
            print('request Error! [Exception]', e, flush=True)
            continue

        # get each repo on this page
        for item in data['items']:
            print(f"{i} -th page, topic = {topic}, repo = {item['full_name']}", flush=True)
            # cnt = cnt + crawl_repo(item)
            cnt = cnt + download_code(item)

    print('-----Topic {' + topic + '} ends: successfully got', cnt, 'repos in total-----\n', flush=True)



def download_worker(response):
    code_info = response.json()
    if "?" in response.url:  # distinguish folder url
        url_info = response.url.split("?")[0].split('/')
        repo_name = url_info[4] + '/' + url_info[5] + "/" + "/".join(url_info[7:])
    else:  # single file
        url_info = response.url.split('/')
        repo_name = url_info[4] + '/' + url_info[5]

    for repo_code in code_info:
        download_url = repo_code['download_url']
        file_name = repo_code['name']
        if not file_name.endswith('.py'): # skip non-python file,
            continue
        if repo_code['type'] == "dir":
            # recursively download files that are in folders
            response = requests.get(repo_code['url'], headers={'Authorization': 'token ' + get_token()})
            download_worker(response)
        if download_url == None:
            continue
        if not os.path.exists(f'{repo_saving_dir}/files/{repo_name}'):
            os.makedirs(f'{repo_saving_dir}/files/{repo_name}')
        # print("download_url", download_url)
        # print("save path", f'{repo_saving_dir}/files/{repo_name}/{file_name}')
        subprocess.run(["wget", "-nc", "-nv", download_url, "--output-document",
                        f'{repo_saving_dir}/files/{repo_name}/{file_name}'])  # skip downloads that would download to existing files (overwriting them)


def download_code(item):
    # check if downloaded it before
    repo_full_name = item['full_name']
    if (repo_full_name in topic_map):
        return 1

    try:
        token_index = random.randint(0, len(tokens) - 1)
        contents_url = 'https://api.github.com/repos/' + repo_full_name + '/contents'
        response = requests.get(contents_url, headers={'Authorization': 'token ' + tokens[token_index]})
        download_worker(response)
    except BaseException as e:
        print('[Exception]', e, flush=True)
        fail_list.append(repo_full_name)
        return 0
    else:
        topic_map[repo_full_name] = item['topics']
        return 1


##################################################
os.system('rm ' + repo_saving_dir + ' -r -f')
topic_map = dict()
##################################################
# Use the code above if you want to delete the previously downloaded repos
# Use the code below if you don't
##################################################
# f = open('./data/github-repos/topic_map.jsonl', 'r')
# topic_map = json.loads(f.readline())
# f.close()
##################################################

for x in topics:
    crawl_topic(x)

f = open(repo_saving_dir + 'topic_map.jsonl', 'w')
json.dump(topic_map, fp = f)
f.close()

f = open(repo_saving_dir + 'fail_list.jsonl', 'w')
json.dump(fail_list, fp = f)
f.close()

print(topic_map, flush=True)
print(fail_list, flush=True)



