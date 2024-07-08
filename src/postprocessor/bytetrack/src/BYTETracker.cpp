#include "BYTETracker.h"
#include <fstream>

BYTETracker::BYTETracker(int frame_rate, int track_buffer)
{
	track_thresh = 0.3; // 和置信度比较的值0.5
	high_thresh = 0.6;
	match_thresh = 0.8;

	frame_id = 0;
	max_time_lost = int(frame_rate / 30.0 * track_buffer);
}

BYTETracker::~BYTETracker()
{
}

vector<STrack> BYTETracker::update(const vector<bytetrackObject> &objects)
{

	////////////////// Step 1: Get detections //////////////////
	this->frame_id++;
	vector<STrack> activated_stracks;
	vector<STrack> refind_stracks;
	vector<STrack> removed_stracks; // 这些向量用于存储不同状态的轨迹
	vector<STrack> lost_stracks;
	vector<STrack> detections;
	vector<STrack> detections_low;

	vector<STrack> detections_cp;
	vector<STrack> tracked_stracks_swap;
	vector<STrack> resa, resb;
	vector<STrack> output_stracks;

	vector<STrack *> unconfirmed;
	vector<STrack *> tracked_stracks;
	vector<STrack *> strack_pool;
	vector<STrack *> r_tracked_stracks;

	if (objects.size() > 0)
	{
		for (int i = 0; i < objects.size(); i++)
		{
			vector<float> tlbr_;
			tlbr_.resize(4);
			tlbr_[0] = objects[i].rect.x;
			tlbr_[1] = objects[i].rect.y;
			tlbr_[2] = objects[i].rect.x + objects[i].rect.width;
			tlbr_[3] = objects[i].rect.y + objects[i].rect.height;

			float score = objects[i].prob;
			STrack strack(STrack::tlbr_to_tlwh(tlbr_), score, objects[i]);

			if (score >= track_thresh)
			{
				detections.push_back(strack);
			}
			else
			{
				detections_low.push_back(strack);
			}
		}
	}

	// Add newly detected tracklets to tracked_stracks
	for (int i = 0; i < this->tracked_stracks.size(); i++)
	{
		if (!this->tracked_stracks[i].is_activated)
			unconfirmed.push_back(&this->tracked_stracks[i]);
		else
			tracked_stracks.push_back(&this->tracked_stracks[i]);
	}

	////////////////// Step 2: First association, with IoU //////////////////

	strack_pool = joint_stracks(tracked_stracks, this->lost_stracks); // 将激活的轨迹和丢失的轨迹合并到 strack_pool 中
	STrack::multi_predict(strack_pool, this->kalman_filter);		  // 对 strack_pool 中的轨迹进行预测

	vector<vector<float>> dists;
	int dist_size = 0, dist_size_size = 0;
	dists = iou_distance(strack_pool, detections, dist_size, dist_size_size); // 计算激活的轨迹与新检测到的目标之间的 IoU（交并比）距离。

	vector<vector<int>> matches;
	vector<int> u_track, u_detection;
	// 使用线性分配算法对距离矩阵进行分配，匹配激活的轨迹与新检测到的目标。
	linear_assignment(dists, dist_size, dist_size_size, match_thresh, matches, u_track, u_detection);
	// 遍历匹配结果，对每个匹配进行处理。
	// 如果轨迹状态为已追踪（Tracked），则更新轨迹信息并将轨迹添加到 activated_stracks 向量中。
	// 如果轨迹状态不为已追踪，表示是重新激活的轨迹，更新轨迹信息并将轨迹添加到 refind_stracks 向量中。
	for (int i = 0; i < matches.size(); i++)
	{
		STrack *track = strack_pool[matches[i][0]];
		STrack *det = &detections[matches[i][1]];
		if (track->state == TrackState::Tracked)
		{
			track->update(*det, this->frame_id);
			activated_stracks.push_back(*track);
		}
		else
		{
			track->re_activate(*det, this->frame_id, false);
			refind_stracks.push_back(*track);
		}
	}

	////////////////// Step 3: Second association, using low score dets //////////////////
	// 目标追踪算法的关联、更新和状态管理步骤，根据匹配结果更新轨迹状态并管理丢失的轨迹。
	for (int i = 0; i < u_detection.size(); i++)
	{
		detections_cp.push_back(detections[u_detection[i]]);
	} // 将未匹配的新检测目标从 detections 向量中复制到 detections_cp 向量中
	detections.clear();
	detections.assign(detections_low.begin(), detections_low.end()); // 将低置信度的目标替换为新的目标。

	for (int i = 0; i < u_track.size(); i++)
	{
		if (strack_pool[u_track[i]]->state == TrackState::Tracked)
		{
			r_tracked_stracks.push_back(strack_pool[u_track[i]]);
		}
	}

	dists.clear();
	dists = iou_distance(r_tracked_stracks, detections, dist_size, dist_size_size); // 计算已追踪的轨迹与未匹配的新检测目标之间的 IoU 距离

	matches.clear();
	u_track.clear();
	u_detection.clear();
	linear_assignment(dists, dist_size, dist_size_size, 0.5, matches, u_track, u_detection);

	for (int i = 0; i < matches.size(); i++)
	{
		STrack *track = r_tracked_stracks[matches[i][0]];
		STrack *det = &detections[matches[i][1]];
		if (track->state == TrackState::Tracked)
		{
			track->update(*det, this->frame_id);
			activated_stracks.push_back(*track);
		}
		else
		{
			track->re_activate(*det, this->frame_id, false);
			refind_stracks.push_back(*track);
		}
	}

	for (int i = 0; i < u_track.size(); i++)
	{
		STrack *track = r_tracked_stracks[u_track[i]];
		if (track->state != TrackState::Lost)
		{
			track->mark_lost();
			lost_stracks.push_back(*track);
		}
	}

	// Deal with unconfirmed tracks, usually tracks with only one beginning frame
	// 将未激活的轨迹与新检测目标进行匹配，更新轨迹状态，并标记被移除的轨迹。
	detections.clear();
	detections.assign(detections_cp.begin(), detections_cp.end());

	dists.clear();
	dists = iou_distance(unconfirmed, detections, dist_size, dist_size_size);

	matches.clear();
	vector<int> u_unconfirmed;
	u_detection.clear();
	linear_assignment(dists, dist_size, dist_size_size, 0.5, matches, u_unconfirmed, u_detection);

	for (int i = 0; i < matches.size(); i++)
	{
		unconfirmed[matches[i][0]]->update(detections[matches[i][1]], this->frame_id);
		activated_stracks.push_back(*unconfirmed[matches[i][0]]);
	}

	for (int i = 0; i < u_unconfirmed.size(); i++)
	{
		STrack *track = unconfirmed[u_unconfirmed[i]];
		track->mark_removed();
		removed_stracks.push_back(*track);
	}

	////////////////// Step 4: Init new stracks //////////////////
	for (int i = 0; i < u_detection.size(); i++)
	{
		STrack *track = &detections[u_detection[i]];
		if (track->score < this->high_thresh)
			continue;
		track->activate(this->kalman_filter, this->frame_id);
		activated_stracks.push_back(*track);
	}

	////////////////// Step 5: Update state //////////////////
	for (int i = 0; i < this->lost_stracks.size(); i++)
	{
		if (this->frame_id - this->lost_stracks[i].end_frame() > this->max_time_lost)
		{
			this->lost_stracks[i].mark_removed();
			removed_stracks.push_back(this->lost_stracks[i]);
		}
	}

	for (int i = 0; i < this->tracked_stracks.size(); i++)
	{
		if (this->tracked_stracks[i].state == TrackState::Tracked)
		{
			tracked_stracks_swap.push_back(this->tracked_stracks[i]);
		}
	}
	this->tracked_stracks.clear();
	this->tracked_stracks.assign(tracked_stracks_swap.begin(), tracked_stracks_swap.end());

	this->tracked_stracks = joint_stracks(this->tracked_stracks, activated_stracks);
	this->tracked_stracks = joint_stracks(this->tracked_stracks, refind_stracks);

	this->lost_stracks = sub_stracks(this->lost_stracks, this->tracked_stracks);
	for (int i = 0; i < lost_stracks.size(); i++)
	{
		this->lost_stracks.push_back(lost_stracks[i]);
	}

	this->lost_stracks = sub_stracks(this->lost_stracks, this->removed_stracks);
	for (int i = 0; i < removed_stracks.size(); i++)
	{
		this->removed_stracks.push_back(removed_stracks[i]);
	}

	remove_duplicate_stracks(resa, resb, this->tracked_stracks, this->lost_stracks);
	this->tracked_stracks.clear();
	this->tracked_stracks.assign(resa.begin(), resa.end());
	this->lost_stracks.clear();
	this->lost_stracks.assign(resb.begin(), resb.end());

	for (int i = 0; i < this->tracked_stracks.size(); i++)
	{
		if (this->tracked_stracks[i].is_activated)
		{
			output_stracks.push_back(this->tracked_stracks[i]);
		}
	}

	// 若removed_stracks.size 大于1000 则删除前200，避免内存泄漏
	if (this->removed_stracks.size() > 1000)
	{
		for (auto it = this->removed_stracks.begin(); this->removed_stracks.size() > 800;)
			it = this->removed_stracks.erase(it);
	}

	return output_stracks;
}