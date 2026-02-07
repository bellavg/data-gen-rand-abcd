import os, sys,random,shutil
import argparse

homeDir = None#os.environ["HOME"]
graphDataFolder = None #os.path.join(homeDir,"OPENABC_DATASET","bench")
scriptsDataFolder = None #os.path.join(homeDir,"OPENABC_DATASET","synScripts")
libraryCellFolder = None #os.path.join(homeDir,"OPENABC_DATASET","lib")
libraryFile = None # Full path to the library file

designSet6 = ['128','256','512','1024','2048','4096','8192','16384']
designs = designSet6
numSynthesizedScript = 1500
delimiter = '\n'

def genShellScriptForSynthesis():
    for des in designs:
        designScriptFile = open(os.path.join(graphDataFolder,'synthesisBulk_'+des+'.sh'),'w+')
        logFolder = os.path.join(graphDataFolder,des,'log_'+des)
        if not os.path.exists(logFolder):
            os.makedirs(logFolder)
        for i in range(numSynthesizedScript):
            synScriptPath = os.path.join(scriptsDataFolder,des,'abc'+str(i)+".script")
            logFilePath = os.path.join(logFolder,'log_'+des+'_syn'+str(i)+'.log')
            synFolder = os.path.join(graphDataFolder,des,'syn'+str(i))
            
            # Create directory for synthesis outputs
            mkdirCmd = 'mkdir -p '+synFolder
            synRunCmd = 'abc -f '+synScriptPath+' > '+logFilePath
            zipCmd = 'zip -q -j -r '+synFolder+'.zip '+synFolder+"/"
            rmCmd = 'rm -fr '+synFolder+"/"
            
            designScriptFile.write(mkdirCmd+delimiter)
            designScriptFile.write(synRunCmd+delimiter)
            designScriptFile.write(zipCmd+delimiter)
            designScriptFile.write(rmCmd+delimiter)
        designScriptFile.close()

def setGlobalAndEnvironmentVars(cmdArgs):
    global homeDir, graphDataFolder,scriptsDataFolder,libraryCellFolder,libraryFile
    homeDir = cmdArgs.home
    if not (os.path.exists(homeDir)):
        print("\nPlease rerun with appropriate paths")
    graphDataFolder = os.path.join(homeDir,"OPENABC_DATASET","bench")
    scriptsDataFolder = os.path.join(homeDir,"OPENABC_DATASET","synScripts")
    libraryCellFolder = os.path.join(homeDir,"OPENABC_DATASET","lib")
    
    # Set library file: use provided path or default to nangate45.lib
    if cmdArgs.lib:
        libraryFile = cmdArgs.lib
    else:
        libraryFile = os.path.join(libraryCellFolder,"nangate45.lib")

def parseCmdLineArgs():
    parser = argparse.ArgumentParser(prog='AUTOMATE SYNTHESIS FLOW', description="Circuit characteristics")
    parser.add_argument('--version',action='version', version='1.0.0')
    parser.add_argument('--home',required=True, help="OpenABC dataset home path")
    parser.add_argument('--lib', required=False, help="Path to library file (default: OPENABC_DATASET/lib/nangate45.lib)")
    return parser.parse_args()

def main():
    cmdArgs = parseCmdLineArgs()
    setGlobalAndEnvironmentVars(cmdArgs)
    genShellScriptForSynthesis()

if __name__ == '__main__':
    main()