"""Build state parity test first; optionally build an isolated engine after gate."""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
BUILD=ROOT/'build-distribution/upstream-media'
OUT=HERE/'build'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--engine',action='store_true');a=p.parse_args()
    OUT.mkdir(exist_ok=True)
    vf={}
    for line in (BUILD/'CMakeFiles/strata_engine.dir/flags.make').read_text().splitlines():
        if ' = ' in line:
            k,v=line.split(' = ',1);vf[k]=shlex.split(v)
    original=(ROOT/'distribution/upstream/src/core/conversation_state.cpp').read_text()
    marker='} // namespace strata::core'
    assert original.count(marker)==1
    source=original.replace(marker,(HERE/'function.cpp.inc').read_text()+'\n'+marker)
    state=OUT/'conversation_state.cpp';state.write_text(source)
    commands=[]
    def compile(src):
        obj=OUT/(src.name+'.o')
        cmd=['/usr/bin/c++',*vf['CXX_DEFINES'],*vf['CXX_INCLUDES'],*vf['CXX_FLAGS'],
             '-I'+str(ROOT/'distribution/upstream/src/core'),'-c',str(src),'-o',str(obj)]
        subprocess.run(cmd,cwd=BUILD,check=True);commands.append(cmd);return obj
    obj=compile(state)
    entry=compile(HERE/'parity.cpp')
    base=[x for x in shlex.split((BUILD/'CMakeFiles/strata.dir/link.txt').read_text()) if not x.startswith('-Wl,--dependency-file=')]
    def link(entry,binary,extra):
        cmd=[str(entry) if x.endswith('generate.cpp.o') else x for x in base]
        cmd[cmd.index('-o')+1]=str(binary)
        i=cmd.index('libstrata_engine.a');cmd[i:i]=[str(x) for x in extra]
        subprocess.run(cmd,cwd=BUILD,check=True);commands.append(cmd)
    test=OUT/'parity';link(entry,test,[obj])
    record=dict(commands=commands,cwd=str(BUILD),source_sha256=sha(state),test_sha256=sha(test))
    if a.engine:
        old=(ROOT/'experimental/native-batch/source/generate.cpp').read_text()
        declaration='''namespace strata::core {
bool conversation_state_transfer(const SessionState&, SessionState&, const ModelGeometry&,
                                 const std::vector<int32_t>&, std::string&);
}
'''
        # All includes precede the first anonymous namespace.
        assert 'namespace {' in old
        new=old.replace('namespace {',declaration+'\nnamespace {',1)
        one='''                strata::core::ConversationCheckpoint ck;
                ck.ids = ids;
                if (!strata::core::conversation_checkpoint_save(ck, from, g, e) ||
                    !strata::core::conversation_checkpoint_restore(ck, to, g, e))'''
        two='''                    strata::core::ConversationCheckpoint ck;
                    ck.ids = bs[(size_t) b].ids;
                    if (!strata::core::conversation_checkpoint_save(ck, from, g, e) ||
                        !strata::core::conversation_checkpoint_restore(ck, to, g, e))'''
        assert new.count(one)==new.count(two)==1
        new=new.replace(one,'                if (!strata::core::conversation_state_transfer(from, to, g, ids, e))')
        new=new.replace(two,'                    if (!strata::core::conversation_state_transfer(from, to, g, bs[(size_t) b].ids, e))')
        src=OUT/'generate.cpp';src.write_text(new)
        (HERE/'entry.patch').write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='a/generate.cpp',tofile='b/generate.cpp')))
        # Native media entry has its own include overlay and compile definitions.
        entry=OUT/'generate.cpp.o'
        cmd=['/usr/bin/c++','-I'+str(ROOT/'experimental/native-batch/source/include'),
             '-DSTRATA_PREFILL_FUSED=1','-DSTRATA_PREFILL_MMQ=1',*vf['CXX_DEFINES'],
             *vf['CXX_INCLUDES'],*vf['CXX_FLAGS'],'-c',str(src),'-o',str(entry)]
        subprocess.run(cmd,cwd=BUILD,check=True);commands.append(cmd)
        verifier=ROOT/'experimental/batch-self-commit/build/verify.cpp.o'
        binary=OUT/'strata-direct-slot-state'
        link(entry,binary,[obj,verifier])
        record.update(binary_sha256=sha(binary),entry_source_sha256=sha(src),verifier_sha256=sha(verifier))
    (OUT/'build.json').write_text(json.dumps(record,indent=2)+'\n')
    print(test,flush=True)
if __name__=='__main__':main()
