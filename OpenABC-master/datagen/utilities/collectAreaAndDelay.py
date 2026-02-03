import os,sys
import re,argparse
import os.path as osp

designSet6 = ['128','256','512','1024','2048','4096','8192','16384']
designs = designSet6

homeDir = None
benchDataFolder = None
statsDataFolder = None

NUM_SYNTHESIZED_DESIGNS = 1500
csvDelimiter = ","

designSet = designs

def getFileLines(filePath):
    f = open(filePath,'r')
    fLines = f.readlines()
    f.close()
    return fLines

def collectAreaAndDelay():
    adpFolder = osp.join(statsDataFolder,"adp")
    if not os.path.exists(adpFolder):
        os.mkdir(adpFolder)
    for des in designs:
        desLogDir = osp.join(benchDataFolder,des,"log_"+des)
        csv_file = os.path.join(adpFolder, 'adp_'+des+'.csv')
        csvFileHandler = open(csv_file,'w+')
        csvFileHandler.write("sid,area,delay\n")
        for i in range(NUM_SYNTHESIZED_DESIGNS):
            synth_stat_file = os.path.join(desLogDir,'log_'+des+"_syn"+str(i)+'.log')
            synthFileLines = getFileLines(synth_stat_file)
            information = re.findall('[a-zA-Z0-9.]+',synthFileLines[-1])
            csvFileHandler.write(str(i)+csvDelimiter+str(information[-9])+csvDelimiter+str(information[-4])+"\n")
        csvFileHandler.close()

def setGlobalAndEnvironmentVars(cmdArgs):
    global homeDir,benchDataFolder,statsDataFolder
    homeDir = cmdArgs.home
    if not (os.path.exists(homeDir)):
        print("\nPlease rerun with appropriate paths")
    benchDataFolder = os.path.join(homeDir,"OPENABC_DATASET","bench")
    statsDataFolder = os.path.join(homeDir,"OPENABC_DATASET","statistics")

def parseCmdLineArgs():
    parser = argparse.ArgumentParser(prog='Final AIG area and delay Collection', description="Circuit characteristics")
    parser.add_argument('--version',action='version', version='1.0.0')
    parser.add_argument('--home',required=True, help="OpenABC dataset home path")
    return parser.parse_args()

def main():
    cmdArgs = parseCmdLineArgs()
    setGlobalAndEnvironmentVars(cmdArgs)
    collectAreaAndDelay()


if __name__ == '__main__':
    main()
