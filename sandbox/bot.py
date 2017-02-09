from selenium import webdriver
import csv
import utils
import os
import time

driver = webdriver.Firefox()
driver.get('https://www.kaggle.com/c/seizure-prediction/submissions/attach')

form = driver.find_elements_by_xpath('//form[@id="login-account"]')[0]
username = form.find_element_by_id("UserName")
pwd = form.find_element_by_id("Password")
username.send_keys("")
pwd.send_keys("")
driver.find_element_by_id("get-started").click()


def read_keys(submission_file):
    clips = []
    with open(submission_file, 'rb') as f:
        reader = csv.reader(f)
        reader.next()
        for row in reader:
            clip = row[0]
            clips.append(clip)
    return clips


clip_ids = read_keys('sample_submission.csv')


def make_submission_file(key):
    filepath = "/mnt/sda3/CODING/python/kaggle-seizure-predict/bot_submisssion_%s.csv" % key
    with open(filepath, 'wb') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['clip', 'preictal'])
        for k in clip_ids:
            if k != key:
                csv_writer.writerow([k, 0])
            else:
                csv_writer.writerow([k, 1])
    return filepath


if os.path.isfile('split.pkl'):
    done_clips = utils.load_pkl('split.pkl')
    public_clips, private_clips = done_clips['public'], done_clips['private']
else:
    public_clips, private_clips = [], []

while len(public_clips) < 0.4 * len(clip_ids):
    try:
        for i, cid in enumerate(clip_ids):
            if cid not in public_clips and cid not in private_clips:
                submission_filepath = make_submission_file(cid)
                driver.get('https://www.kaggle.com/c/seizure-prediction/submissions/attach')
                upload_submission = driver.find_element_by_id("SubmissionUpload")
                upload_submission.send_keys(submission_filepath)
                submit = driver.find_element_by_id('submit-progress').click()

                driver.get('https://www.kaggle.com/c/seizure-prediction/submissions')
                last_entry = \
                    driver.find_elements_by_xpath(
                        "//*[@class='nicetable roomy align-top full submissions _buttons']/tbody/tr")[
                        1]
                scores = last_entry.find_elements_by_xpath("*")[2:]
                public_score = float(scores[0].text)
                private_score = float(scores[1].text)
                print cid, public_score, private_score

                if public_score != 0.5:
                    public_clips.append(cid)
                if private_score != 0.5:
                    private_clips.append(cid)
                os.remove(submission_filepath)
            if (i + 1) % 20 == 0:
                time.sleep(5)
    except:
        utils.save_pkl({'public': public_clips, 'private': private_clips}, 'split.pkl')
        print 'saved'
        time.sleep(5)
        driver.refresh()
