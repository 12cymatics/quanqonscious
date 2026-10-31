import {readFileSync,writeFileSync,mkdirSync} from 'node:fs';
import {resolve} from 'node:path';
import {generateGallery} from './render_water_sand_gallery.mjs';
import {patternCase} from './pattern_case.mjs';
import {commonWaterWindow} from './water_common_window.mjs';
const out=resolve('water-sweep-report');mkdirSync(out,{recursive:true});
const rows=[],inputs=[],records=[];
const read=(path)=>{try{return JSON.parse(readFileSync(path,'utf8'));}catch{return null;}};
for(const volume of ['1.5','2.5'])for(const accel of ['0.5','1'])for(const contact of ['free','pinned']){
 const id=`water-case-v${volume}-a${accel}-${contact}`,root=resolve('all-cases',id),dir=resolve(root,'case-output');
 const config=read(resolve(dir,'config.json')),summary=read(resolve(dir,'summary.json')),audit=read(resolve(root,'ns-pattern-audit/audit.json'));
 inputs.push({dir,label:`${volume} mL / ${accel} g / ${contact}`});
 const row={id,planned:{frequencyHz:56,volumeMl:Number(volume),peakAccelerationG:Number(accel),contact,cycles:72,wallCapSeconds:3600},
  outcome:summary?.status||'missing-output',error:summary?.error||null,actualCycles:summary?.simulatedDriveCycles??null,
  audit: audit?.validation||'not available',auditFailure:audit?.failure||null,config,summary};
 const expected=patternCase({PATTERN_VOLUME_ML:volume,PATTERN_ACCEL_G:accel,PATTERN_CONTACT:contact}).options;
 row.configMatchesPlan=!!config&&config.frequency===56&&['R','h','accel','contact','nr','nth','nz'].every(key=>config.options?.[key]===expected[key])&&config.seed===1729&&config.seedRms===1e-7;
 row.equalHorizonEligible=row.configMatchesPlan&&row.outcome==='completed-horizon'&&row.actualCycles>=72-1e-10&&audit?.validation==='saved fields finite and positive-depth; accepted final velocity divergence recomputed, not threshold-certified';
 let raw=null;try{raw=readFileSync(resolve(dir,'surface.ndjson'),'utf8').trim().split('\n').map(JSON.parse);}catch{}
 records.push({id,planned:row.planned,config,summary,audit,grid:read(resolve(dir,'grid.json')),raw});
 rows.push(row);
}
const hashes=[...new Set(rows.filter(r=>r.config).map(r=>r.config.initialSurfaceSha256))];
const controlsConsistent=hashes.length===1&&hashes[0]!==undefined&&rows.filter(r=>r.config).every(r=>r.config.initialCondition==='common-double-zero-rim-broadband-v1');
const gallery=generateGallery(inputs,out);
const commonWindow=commonWaterWindow(records);
if(commonWindow.available){
 const commonInputs=inputs.filter((input,i)=>commonWindow.includedIds.includes(rows[i].id));
 commonWindow.gallery=generateGallery(commonInputs,resolve(out,'common-window'),{requestedCycle:commonWindow.commonCycle});
}
writeFileSync(resolve(out,'common-window.json'),JSON.stringify(commonWindow,null,2));
const report={schema:'water-ns-factorial-exploration/1',commit:process.env.GITHUB_SHA,run:process.env.GITHUB_RUN_ID,
 design:'56Hz;1.5/2.5mL;0.5/1g peak assumed drive;free/pinned ideal contact;full2x2x2factorial;72drivecycles;max2parallel',
 commonWindow,controlsConsistent,initialSurfaceHashes:hashes,cases:rows,galleryCases:gallery.cases,
 limitations:['Finite16x48x16 numerical exploration, not spatial/time converged pattern selection.',
 'Initial field differs from historical0.5g/1g pilots; old illustrations are not strict controls.',
 'No selected mode was seeded; all planned outcomes are retained, including missing/failed/partial.',
 'Amplitude is specified simulation forcing, not inferred from experimental mV.',
 'Sand-like stippling illustrates signed water elevation; no grain transport or optical photography model.',
 'Two temporal samples are one drive period apart; opposite water phase is not assumed.',
 '72cycles is below retained128cycle establishment gate; no saturated-pattern certification.']};
writeFileSync(resolve(out,'all-outcomes.json'),JSON.stringify(report,null,2));
const lines=['# Water variation batch','',report.design,'','| Volume mL | Peak g | Contact | Outcome | Cycles | Audited equal horizon |','|---|---|---|---|---:|---|'];
for(const r of rows)lines.push(`|${r.planned.volumeMl}|${r.planned.peakAccelerationG}|${r.planned.contact}|${r.outcome}|${r.actualCycles??'unknown'}|${r.equalHorizonEligible}|`);
lines.push('','## Exploratory shorter common window',commonWindow.available?`Common cycle ${commonWindow.commonCycle}; included ${commonWindow.includedIds.length} of ${rows.length} cases. This does not complete the requested 72 cycles.`:commonWindow.reason,`Excluded cases: ${commonWindow.excludedIds.join(', ')||'none'}`);
lines.push('','## Limits',...report.limitations.map(x=>'- '+x));writeFileSync(resolve(out,'README.md'),lines.join('\n')+'\n');
console.log(JSON.stringify({controlsConsistent,cases:rows.map(r=>({id:r.id,outcome:r.outcome,actualCycles:r.actualCycles,equalHorizonEligible:r.equalHorizonEligible}))},null,2));
if(!controlsConsistent||rows.some(r=>!r.equalHorizonEligible))process.exitCode=1;
