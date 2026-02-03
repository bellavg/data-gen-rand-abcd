import networkx as nx
#import dgl
import argparse,os,re
import pandas as pd
import os.path as osp

homeDir = None#os.environ["HOME"]
graphDataFolder = None #os.path.join(homeDir,"OPENABC_DATASET","bench")
statsDataFolder = None #os.path.join(homeDir,"OPENABC_DATASET","synScripts")

designSet6 = ['128','256','512','1024','2048','4096','8192','16384']
designs = designSet6

def collectFinalStats():
    global graphDataFolder,statsDataFolder
    designScriptFile = open(os.path.join(graphmlDataFolder,'collectFinalAIGData.sh'),'w+')
    designScriptFile.write("#!/bin/bash\n")
    for des in designs:
        cmd = 'python collectGraphStatistics.py --gml '+osp.join(graphmlDataFolder,des)+" --des "+des+" --stats "+statsDataFolder
        designScriptFile.write(cmd+"\n")
    designScriptFile.close()

def setGlobalAndEnvironmentVars(cmdArgs):
    global homeDir,benchDataFolder,statsDataFolder,graphmlDataFolder
    homeDir = cmdArgs.home
    if not (os.path.exists(homeDir)):
        print("\nPlease rerun with appropriate paths")
    benchDataFolder = os.path.join(homeDir,"OPENABC_DATASET","bench")
    graphmlDataFolder = os.path.join(homeDir,"OPENABC_DATASET","graphml")
    statsDataFolder = os.path.join(homeDir,"OPENABC_DATASET","statistics")

def parseCmdLineArgs():
    parser = argparse.ArgumentParser(prog='AUTOMATE SYNTHESIS FLOW', description="Circuit characteristics")
    parser.add_argument('--version',action='version', version='1.0.0')
    parser.add_argument('--home',required=True, help="OpenABC dataset home path")
    return parser.parse_args()

def main():
    cmdArgs = parseCmdLineArgs()
    setGlobalAndEnvironmentVars(cmdArgs)
    collectFinalStats()

if __name__ == '__main__':
    main()