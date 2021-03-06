# formulate pred generated by predictor
import sys
import json
import pdb
from copy import deepcopy
import csv

def read_domain_slot_list(filename):
  with open(filename) as fp:
    lines = fp.readlines()
  domain_slots = []
  for line in lines:
    if line.startswith("#"):
      continue
    if len(line.strip("\n ")) == 0 :
      continue
    line_arr = line.split("\t")
    ds = line_arr[0] + " " + line_arr[1]
    if line_arr[3] == "n":
      domain_slots.append(ds)
  return domain_slots

fp = open(sys.argv[1])
lines = fp.readlines()

dialogs = []
js = {}
for line in lines:
  line = line.strip("\n")
  if line[:5] != "input" and line[:10] != "prediction":
    continue
  if line[:5] == "input":
    js = json.loads(line[line.find(":")+1:])
  if line[:10] == "prediction":
    prediction = json.loads(line[line.find(":")+1:])
    dialogs.append((js, prediction))

def calc_pred_belief_state(prediction, ds_list, ontology):
  def dict2str(d):
    res = []
    for k, v in d.items():
      res.append(k+":"+v)
    return sorted(res)

  prediction = prediction["predicted_labels"]
  turn_bs = []
  for turn in prediction:
    cur_bs = {}
    for ds in ds_list:
      if ds not in ontology: continue
      cur_bs[ds] = "none"
    for slot_value in turn:
      p = slot_value.find(":")
      slot = slot_value[:p]
      if slot not in ontology: continue
      value = slot_value[p+1:] # value may have ":"
      cur_bs[slot] = value
    turn_bs.append(dict2str(cur_bs))
  return turn_bs
   
def calc_acc(true_labels, pred_labels):
  assert(len(true_labels) == len(pred_labels))
  total_turn = 0.0
  err_turn = 0.0
  wrong_dialog = []
  for d in range(len(true_labels)): # for each dialog
    err_of_dialog = 0
    assert(len(true_labels[d]) == len(pred_labels[d]))
    for t in range(len(true_labels[d])): # for each turn
      total_turn += 1
      if len(true_labels[d][t]) != len(pred_labels[d][t]):
        err_turn += 1
        err_of_dialog += 1
        continue
      for x, y in zip(true_labels[d][t], pred_labels[d][t]):
        if x != y:
          err_turn += 1
          err_of_dialog += 1
          break
    if err_of_dialog > 0:
      wrong_dialog.append(d)
  return (total_turn - err_turn) / total_turn, wrong_dialog

def slot_acc(true_labels, pred_labels):
  assert(len(true_labels) == len(pred_labels))
  total_slots = 0.0
  err_slots = 0.0
  wrong_labels = []
  for d in range(len(true_labels)): # for each dialog
    err_of_dialog = 0
    assert(len(true_labels[d]) == len(pred_labels[d]))
    for t in range(len(true_labels[d])): # for each turn
      num_turn_slots = min(len(true_labels[d][t]),len(pred_labels[d][t]))
      total_slots += num_turn_slots
      for idx in range(num_turn_slots):
        if true_labels[d][t][idx] != pred_labels[d][t][idx]:
          err_slots += 1
          wrong_labels.append((true_labels[d][t][idx],pred_labels[d][t][idx]))
  return (total_slots - err_slots) / total_slots, wrong_labels


ds_list = read_domain_slot_list("./ontology/domain_slot_list_sp.txt")# Change if no spans
ontology = set(ds_list)
true_labels = []
pred_labels = []
for dialog, prediction in dialogs:
  dialog_bs = []
  for turn in dialog["dialogue"]:
    turn_bs = []
    ds_set = set(ds_list)
    for domain, v in turn["belief_state"].items():
      for slot, slot_value in v["semi"].items():
        ds = domain + " " + slot
        if ds not in ontology:
          continue
        if slot_value == "": slot_value = "none"
        turn_bs.append(domain + " " + slot + ":" + slot_value)
        ds_set.remove(domain + " " + slot)
    for ds in ds_set:
      if ds not in ontology: continue
      turn_bs.append(ds+":"+"none")
    turn_bs = sorted(turn_bs)
    dialog_bs.append(turn_bs)
  true_labels.append(dialog_bs)
  pred_labels.append(calc_pred_belief_state(prediction, ds_list, ontology))

acc, wrong_dialogs = calc_acc(true_labels, pred_labels)
acc_slot, wrong_labels = slot_acc(true_labels, pred_labels)
print(sys.argv[1]+"\t"+"joint_acc: {}".format(acc)+"\t"+"slot_acc: {}".format(acc_slot))

