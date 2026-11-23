import {mkdtempSync,rmSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {join,dirname} from 'node:path';
import {fileURLToPath} from 'node:url';
import {spawnSync} from 'node:child_process';
const directory=dirname(fileURLToPath(import.meta.url));
const scratch=mkdtempSync(join(tmpdir(),'faraday-resolved-gap-join-'));
try{
  const executable=join(scratch,'check-resolved-gap-join');
  const build=spawnSync(process.env.CXX||'c++',['-std=c++20','-O2','-Wall','-Wextra','-Werror','-pedantic',join(directory,'resolved-gap-flow.cpp'),join(directory,'resolved-gap-join.cpp'),join(directory,'check-resolved-gap-join.cpp'),'-o',executable],{encoding:'utf8'});
  if(build.error||build.status!==0)throw new Error(build.error?.message||build.stderr||'Join compilation failed');
  const result=spawnSync(executable,[],{encoding:'utf8'});if(result.stdout)process.stdout.write(result.stdout);
  if(result.error||result.status!==0)throw new Error(result.error?.message||result.stderr||'Conforming NS join check failed');
  const reports=result.stdout.trim().split('\n').map(JSON.parse);if(!reports.at(-1)?.ok)throw new Error('Joined NS check did not pass');
}finally{rmSync(scratch,{recursive:true,force:true});}
