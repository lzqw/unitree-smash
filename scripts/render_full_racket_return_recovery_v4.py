#!/usr/bin/env python3
"""Replay one actual legal air return/recovery and its failed paired TRAIN case."""
import argparse,json
from datetime import datetime,timezone
from pathlib import Path
from render_full_racket_r3_effect import Scene,ROOT,sha
from render_current_v4_progress import camera
import imageio.v2 as imageio
import numpy as np
from PIL import Image,ImageDraw,ImageFont


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--preview-only',action='store_true');a=p.parse_args()
    run,output=a.run.resolve(),a.output.resolve()
    if output.exists() and not a.preview_only:raise ValueError('Use a fresh output')
    output.parent.mkdir(parents=True,exist_ok=True)
    rows=json.loads((run/'episodes_000.json').read_text());events=json.loads((run/'contacts_000.json').read_text())
    good,bad=rows[12],rows[13]
    event=next(x for x in events if x['env_id']==12 and x['episode_index']==0 and x['first_racket_contact'])
    data=np.load(run/'all_first_episodes_000.npz');court=json.loads((ROOT/'configs/court_net_v4.json').read_text())
    assert good['airborne_string_hit'] and good['recovery_success'] and good['crossed_net'] and good['landed_legal_court']
    assert not good['complete_smash'] and not good['landed_target'] and not bad['had_contact']
    assert bad['training_terminal_reasons']['body_ground'] and all(x['episode_index']==0 for x in rows)
    assert good['jump']['takeoff_time_s']<good['contact_time_s']<good['jump']['landing_time_s']
    assert data['valid_first_episode'][:,12].all() and np.isfinite(data['link_poses']).all()
    scene=Scene(run,court)
    for name,w,h,eye,target,fov in [
        ('wide_side',760,480,[1.3,-6.,3.],[1.3,0.,1.],49),
        ('wide_top',504,480,[1.35,-.001,5.5],[1.35,0.,0.],60)]:
        scene.views[name]=(scene.ctx.simple_framebuffer((w,h),components=3),camera(eye,target,w/h,fov))
    font='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    title,text,small=[ImageFont.truetype(font,n) for n in [30,23,20]]
    speed=float(np.linalg.norm(good['outgoing_velocity_mps']));miss=abs(good['first_landing_m'][1])-1.
    fps,frame_count,segments,previews=50,0,[],[]
    ids_good=np.flatnonzero(data['valid_first_episode'][:,12]);ids_bad=np.flatnonzero(data['valid_first_episode'][:,13])
    times=data['episode_clock_s'][:,12]

    def frame(env_id=12,index=None,*,slow=False,exact=False,terminal=False,landing_freeze=False,wide=None):
        row=good if env_id==12 else bad;landing=trail=None
        if exact:
            poses=np.asarray(event['body_link_poses_local']);ball=np.asarray(event['ball_position_m']);t=event['time_s']
        elif terminal:
            state=row['terminal_native_state'];t=state['time_s']
            poses=np.column_stack((state['body_link_positions_m'],state['body_link_quaternions_wxyz']))
            ball=None
        else:
            assert data['valid_first_episode'][index,env_id]
            poses=data['link_poses'][index,env_id];t=float(data['episode_clock_s'][index,env_id])
            ball=data['ball_position_m'][index,env_id] if t>=2 else None
            if env_id==12 and t>=good['first_landing_time_s']:ball=None
            if t>=2:
                chosen=np.arange(max(0,index-25),index+1)
                mask=data['valid_first_episode'][chosen,env_id]&(data['episode_clock_s'][chosen,env_id]>=2)
                if env_id==12:mask&=data['episode_clock_s'][chosen,env_id]<=good['first_landing_time_s']
                trail=data['ball_position_m'][chosen[mask],env_id]
        if env_id==12 and t>=good['first_landing_time_s']:landing=np.asarray(good['first_landing_m'])
        if wide is None:wide=env_id==12 and 2.84<=t<=4.12
        views=['wide_side','wide_top'] if wide else ['side','top']
        image=Image.new('RGB',(1280,720),(18,28,43))
        for view,left in zip(views,[0,776]):
            pane=scene.render(poses,view,ball=ball,trail=trail,landing=landing)
            if wide:
                x0,x1=court['target_x_limits_m'];y=court['target_y_half_width_m']
                corners=[scene.project(point,view) for point in [(x0,-y,.002),(x1,-y,.002),(x1,y,.002),(x0,y,.002)]]
                if all(point is not None for point in corners):
                    d=ImageDraw.Draw(pane);d.line(corners+[corners[0]],fill=(100,245,165),width=2)
                    label=scene.project([(x0+x1)/2,0.,.002],view)
                    if label:d.text((label[0]-45,label[1]-10),'原目标区',font=small,fill=(145,255,190))
            if exact:
                point=scene.project(event['point_m'],view)
                if point:
                    x,y=point;d=ImageDraw.Draw(pane)
                    d.ellipse((x-12,y-12,x+12,y+12),outline=(255,245,80),width=3)
                    d.text((max(8,x-90),max(35,y-40)),'实际接触点',font=small,fill=(255,245,80))
            if landing_freeze:
                point=scene.project(good['first_landing_m'],view)
                if point:
                    x,y=point;ImageDraw.Draw(pane).text((max(8,min(x-130,pane.width-260)),max(40,y-45)),
                        '真实首次落点',font=small,fill=(255,150,120))
            image.paste(pane,(left,106))
        d=ImageDraw.Draw(image)
        heading='同一回合｜空中触球 → 越网回球 → 落地恢复'
        subtitle='原生 TRAIN 1 / 候选6｜回球与恢复已连上，完整扣杀仍失败'
        if env_id==13:
            heading='失败对照｜同一候选的另一训练来球'
            subtitle='原生 TRAIN 65｜未触球，身体触地后终止；未验证两来球泛化'
        elif exact:heading='己方合法空中拍线触球｜原生接触子步定格'
        elif landing_freeze:heading='真实首次落点｜对方场内，横向偏出指定目标'
        d.text((20,10),heading,font=title,fill='white');d.text((20,57),subtitle,font=text,fill=(187,214,231))
        d.text((18,113),'侧视｜己方 ← 网 → 对方'+('｜扩大球路视野' if wide else ''),font=small,fill='white')
        d.text((790,113),'俯视｜同一时刻',font=small,fill='white')
        mode='原生接触子步定格' if exact else ('终止前原生状态定格' if terminal else ('0.25倍慢放' if slow else '正常速度'))
        if landing_freeze:mode='首次落点事件 + 相邻50Hz全身采样'
        d.rectangle((16,548,754,584),fill=(18,28,43));d.text((24,552),f't = {t:.4f} s｜{mode}',font=small,fill='white')
        if env_id==13:
            status='来球与自主动作执行中' if not terminal else f"未触球｜身体触地，t={bad['terminal_native_state']['time_s']:.2f} s终止"
            note='这是另一独立回合的失败；终止之后不补造动作或球路'
        elif landing_freeze:
            status=f"首落地 t={good['first_landing_time_s']:.4f} s｜x={good['first_landing_m'][0]:.3f}, y={good['first_landing_m'][1]:.3f} m"
            note=f'红叉为实际首次事件；偏出目标横向{miss*100:.1f} cm。全身姿态采样 t={t:.2f} s'
        elif exact:
            status=f"腾空拍线触球高 {good['contact_point_m'][2]:.3f} m｜实际出球 {speed:.2f} m/s"
            note='原要求出球≥8 m/s、向下≥1 m/s；本例向下仅0.25 m/s'
        elif t<2:
            status='从地面自主准备和走位';note='完整Go2、标准PiPER及675mm球拍；无悬空外力或回合内重置'
        elif t<good['contact_time_s']:
            status='实际起跳与来球拦截';note='本例合格腾空0.412 s，触球处于同一次起跳的腾空区间'
        elif t<good['crossing_time_s']:
            status=f'真实拍线接触已发生｜实际出球 {speed:.2f} m/s';note='橙线为记录的球心轨迹，球触网记录属于本回合真实事件'
        elif t<good['first_landing_time_s']:
            status='已实际越网｜球有触网记录，未被挡回';note='球触网后合法越网不等于机器人触网；本例机器人触网故障为0'
        else:
            status=f"同回合完成合格恢复｜最终连续稳定 {good['jump']['recovery_s']:.2f} s｜身体触地0"
            note=f'首球落在对方场内，但横向偏出目标{miss*100:.1f} cm；落地后的球滚动不再显示'
        d.text((20,598),status,font=text,fill=(255,205,131));d.text((20,640),note,font=small,fill=(211,225,234))
        d.text((20,681),'本代64次：7空中拍线，1次触球后合格恢复，0完整扣杀｜尚未正式评测',font=small,fill=(163,188,204))
        return image

    landing_i=int(ids_good[np.argmin(np.abs(times[ids_good]-good['first_landing_time_s']))])
    for name,picture in [('contact',frame(exact=True)),('landing',frame(index=landing_i,landing_freeze=True,wide=True)),
                         ('recovery',frame(terminal=True)),('paired_failure',frame(13,terminal=True))]:
        path=output.with_name(output.stem+'_'+name+'.jpg');picture.save(path,quality=94);previews.append(str(path.relative_to(ROOT)))
    if a.preview_only:
        print(json.dumps(dict(previews=previews,renderer=scene.ctx.info['GL_RENDERER']),ensure_ascii=False),flush=True);return
    def card(heading,lines):
        picture=Image.new('RGB',(1280,720),(18,28,43));d=ImageDraw.Draw(picture)
        d.text((50,100),heading,font=title,fill='white');d.line((50,175,1230,175),fill=(73,187,168),width=3)
        for i,line in enumerate(lines):d.text((50,225+i*80),line,font=text,fill=(214,229,238))
        return picture
    with imageio.get_writer(output,fps=fps,codec='libx264',quality=8,pixelformat='yuv420p',
        ffmpeg_params=['-movflags','+faststart','-threads','2']) as writer:
        def still(picture,seconds,name,**extra):
            nonlocal frame_count
            begin=frame_count
            for _ in range(round(seconds*fps)):writer.append_data(np.asarray(picture));frame_count+=1
            segments.append(dict(name=name,start_frame=begin,frames=frame_count-begin,**extra))
        def clip(env_id,ids,slow=False,wide=None,name=None):
            nonlocal frame_count
            begin=frame_count
            for index in ids:
                picture=np.asarray(frame(env_id,int(index),slow=slow,wide=wide))
                for _ in range(4 if slow else 1):writer.append_data(picture);frame_count+=1
            segments.append(dict(name=name,start_frame=begin,frames=frame_count-begin,env_id=env_id,generation=0,
                episode_index=0,recorded_start_s=float(data['episode_clock_s'][ids[0],env_id]),
                recorded_end_s=float(data['episode_clock_s'][ids[-1],env_id]),interpolated=False))
            print('Rendered '+str(name),flush=True)
        still(card('当前进展｜一个真实回合完成空中回球与恢复',[
            '自主起跳、己方拍线触球、球触网后越网、对方场内首次落地。',
            '同回合完成合格恢复，连续稳定3.62秒；完整扣杀仍未成功。',
            '出球仅4.83 m/s、向下0.25 m/s，首次落点横向偏出目标6.7 cm。']),2,'opening')
        clip(12,ids_good,name='same_episode_normal')
        clip(12,ids_good[(times[ids_good]>=2.4)&(times[ids_good]<=3.04)],slow=True,wide=False,name='air_contact_quarter_speed')
        still(frame(exact=True),2.5,'exact_native_contact',time_s=good['contact_time_s'])
        clip(12,ids_good[(times[ids_good]>=2.78)&(times[ids_good]<=4.04)],slow=True,wide=True,name='overnet_landing_quarter_speed')
        still(frame(index=landing_i,landing_freeze=True,wide=True),2.5,'first_landing',event_time_s=good['first_landing_time_s'],pose_time_s=float(times[landing_i]))
        still(frame(terminal=True),2,'qualified_recovered_terminal_state',time_s=8.)
        clip(13,ids_bad,name='paired_case_failure')
        still(frame(13,terminal=True),1.5,'paired_case_body_ground_terminal',time_s=bad['terminal_native_state']['time_s'])
        still(card('完整成功仍为0｜继续优化攻击出球、落点和泛化',[
            '本例已连上合法空中拍线、实际越网与同回合合格恢复。',
            '速度和向下分量仍未达到原攻击标准，目标横向偏出6.7 cm。',
            '同一候选的另一训练来球失败；单例不代表正式两面板成绩。']),2.5,'result')
    protocol=json.loads((ROOT/'configs/g07_full_racket_eval_v4_r3.json').read_text())
    sources=[run/n for n in ['episodes_000.json','contacts_000.json','all_first_episodes_000.npz','population_000.pt',
        'geometry.json','geometry.npz','source_hashes.json','source_snapshot.zip']]
    sources += [Path(__file__),ROOT/'scripts/render_full_racket_r3_effect.py',ROOT/'scripts/render_current_v4_progress.py',
        ROOT/'configs/court_net_v4.json',ROOT/'configs/g07_full_racket_eval_v4_r3.json',ROOT/protocol['asset']['path']]
    assert sha(ROOT/protocol['asset']['path'])==protocol['asset']['sha256']
    metadata=dict(created_at=datetime.now(timezone.utc).isoformat(),video=str(output.relative_to(ROOT)),bytes=output.stat().st_size,
        sha256=sha(output),fps=fps,frames=frame_count,duration_s=frame_count/fps,resolution=[1280,720],
        renderer=scene.ctx.info['GL_RENDERER'],renderer_uses_gpu=False,segments=segments,previews=previews,
        native_recorded_motion=True,interpolated_or_generated_motion=False,main_return_recovery_is_same_episode=True,
        paired_failure_is_separate_episode=True,whole_gpu_job_complete=False,formal_acceptance=False,
        result=dict(airborne_strings=True,crossed_net=True,qualified_recovery=True,stable_recovery_s=good['jump']['recovery_s'],
            actual_outgoing_speed_mps=speed,actual_outgoing_velocity_mps=good['outgoing_velocity_mps'],
            first_landing_m=good['first_landing_m'],target_lateral_miss_m=miss,complete_smash=False),
        caveats=['TRAIN1 candidate6 is a selected component milestone, not a full smash or the worst-case optimizer winner.',
            'Paired TRAIN65 case failed. Formal A/B200 each remains unrun.',
            'Only contact freeze is an exact native substep body snapshot. Landing freeze shows adjacent50Hz body sampling and exact native first-landing event marker.',
            'Ball after its first landing is hidden; later rolling cannot count again.',
            'Orange/yellow rings and red first-landing cross are annotations, not changed physics geometry.',
            'Configured leg speed limit30rad/s; this episode recorded30.002182rad/s maximum solver overshoot, retained in raw metrics.'],
        sources={str(path.relative_to(ROOT)):sha(path) for path in sources})
    output.with_suffix('.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:metadata[k] for k in ['video','bytes','sha256','frames','duration_s']},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
