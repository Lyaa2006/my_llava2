import unittest
from pathlib import Path
import sys

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llava.model.llava_arch import LlavaMetaForCausalLM


class _VisionConfig:
    image_size = 8
    patch_size = 2


class _VisionTower:
    config = _VisionConfig()


class _SpectralHarness(LlavaMetaForCausalLM):
    def __init__(self):
        self.relation_routing_config = {
            "use_spectral_image_routing": True,
            "use_text_anchor_routing": True,
            "spectral_cutoff": 0.33,
            "spectral_low_bins": 4,
            "spectral_high_bins": 4,
            "spectral_image_weight": 0.5,
            "text_weight": 0.5,
            "history_weight": 0.15,
            "routing_temperature": 0.1,
            "routing_min_similarity": -1.0,
            "routing_role_prior_weight": 0.1,
            "routing_role_member_weight": 0.5,
            "routing_role_size_penalty": 0.2,
            "role_member_top_k": 2,
            "routing_early_uniform_mix": 0.25,
            "routing_middle_temperature": 0.5,
            "routing_middle_role_gamma": 1.15,
            "routing_middle_role_uniform_mix": 0.10,
            "routing_middle_task_uniform_mix": 0.10,
            "routing_middle_role_margin_low": 0.10,
            "routing_middle_role_margin_high": 0.30,
            "routing_middle_intra_margin_low": 0.05,
            "routing_middle_intra_margin_high": 0.20,
        }
        self._vision_tower = _VisionTower()
        self.max_task_slots = 4
        self.max_role_slots = 4
        self.expert_num = 4
        self.image_anchors = nn.ParameterList(
            [nn.Parameter(torch.zeros(1, 768)) for _ in range(self.max_task_slots)]
        )
        self.text_anchors = nn.ParameterList(
            [nn.Parameter(torch.zeros(1, 768)) for _ in range(self.max_task_slots)]
        )
        self.text_boundary = nn.ParameterList(
            [
                nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
                for _ in range(self.max_task_slots)
            ]
        )
        self.image_boundary = nn.ParameterList(
            [
                nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
                for _ in range(self.max_task_slots)
            ]
        )
        self.spectral_image_boundary = nn.ParameterList(
            [
                nn.Parameter(torch.tensor([0.0], dtype=torch.float32))
                for _ in range(self.max_task_slots)
            ]
        )
        self.spectral_image_dim = 768 * (
            self.relation_routing_config["spectral_low_bins"]
            + self.relation_routing_config["spectral_high_bins"]
        )
        self.spectral_image_anchors = nn.ParameterList(
            [nn.Parameter(torch.zeros(1, self.spectral_image_dim)) for _ in range(self.max_task_slots)]
        )
        self.role_image_prototypes = nn.ParameterList(
            [nn.Parameter(torch.zeros(1, 768)) for _ in range(self.max_role_slots)]
        )
        self.role_spectral_prototypes = nn.ParameterList(
            [nn.Parameter(torch.zeros(1, self.spectral_image_dim)) for _ in range(self.max_role_slots)]
        )
        self.role_text_prototypes = nn.ParameterList(
            [nn.Parameter(torch.zeros(1, 768)) for _ in range(self.max_role_slots)]
        )
        self.expert_usage_prior = nn.Parameter(torch.zeros(self.max_task_slots))
        self.role_task_count = nn.Parameter(torch.zeros(self.max_role_slots))
        self.role_usage_prior = nn.Parameter(torch.zeros(self.max_role_slots))
        self.task_role_membership = nn.Parameter(
            torch.zeros(self.max_task_slots, self.max_role_slots)
        )
        self.active_role_count = nn.Parameter(torch.zeros(1))

    def get_model(self):
        return self

    def get_vision_tower(self):
        return self._vision_tower


class SpectralRoutingTest(unittest.TestCase):
    def test_descriptor_shape_finite_and_batch_independence(self):
        harness = _SpectralHarness()
        patches = torch.randn(2, 16, 768)

        descriptors = harness._extract_image_spectral_descriptor(patches)
        first = harness._extract_image_spectral_descriptor(patches[:1])
        second = harness._extract_image_spectral_descriptor(patches[1:])

        self.assertEqual(tuple(descriptors.shape), (2, harness.spectral_image_dim))
        self.assertEqual(descriptors.dtype, torch.float32)
        self.assertTrue(torch.isfinite(descriptors).all())
        self.assertTrue(torch.allclose(descriptors[:1], first, atol=1e-5))
        self.assertTrue(torch.allclose(descriptors[1:], second, atol=1e-5))

    def test_radial_masks_cover_frequency_grid_without_overlap(self):
        harness = _SpectralHarness()
        low_bins, high_bins, low, high = harness._build_radial_frequency_masks(
            24, 24, torch.device("cpu")
        )

        self.assertEqual(tuple(low_bins.shape), (4, 24, 24))
        self.assertEqual(tuple(high_bins.shape), (4, 24, 24))
        self.assertTrue(torch.equal(low | high, torch.ones_like(low)))
        self.assertFalse(torch.any(low & high))
        self.assertTrue(torch.equal(low_bins.any(dim=0), low))
        self.assertTrue(torch.equal(high_bins.any(dim=0), high))

    def test_eval_only_counts_contiguous_experts_with_required_anchors(self):
        harness = _SpectralHarness()
        harness.text_boundary[0].data.fill_(5.0)
        harness.text_boundary[1].data.fill_(6.0)
        harness.text_boundary[2].data.fill_(7.0)
        harness.spectral_image_boundary[0].data.fill_(4.0)
        harness.spectral_image_boundary[1].data.fill_(3.0)
        harness.spectral_image_boundary[2].data.fill_(0.0)

        self.assertEqual(harness._get_available_eval_expert_count(), 2)
        self.assertTrue(harness._spectral_image_bank_available(2))
        self.assertFalse(harness._spectral_image_bank_available(3))

    def test_eval_requires_trained_text_anchor_when_text_routing_enabled(self):
        harness = _SpectralHarness()
        harness.spectral_image_boundary[0].data.fill_(4.0)
        harness.spectral_image_boundary[1].data.fill_(3.0)
        harness.text_boundary[0].data.fill_(5.0)
        harness.text_boundary[1].data.fill_(1.0)

        self.assertEqual(harness._get_available_eval_expert_count(), 1)

    def test_mix_with_uniform_preserves_secondary_mass(self):
        harness = _SpectralHarness()
        weights = torch.tensor([0.999, 0.001], dtype=torch.float32)

        mixed = harness._mix_with_uniform(weights, 0.25)

        self.assertGreater(mixed[1].item(), weights[1].item())
        self.assertLess(mixed[0].item(), weights[0].item())
        self.assertAlmostEqual(float(mixed.sum().item()), 1.0, places=6)

    def test_role_build_stage_uses_spectral_image_anchor(self):
        harness = _SpectralHarness()
        harness.spectral_image_boundary[0].data.fill_(4.0)
        harness.spectral_image_boundary[1].data.fill_(4.0)
        harness.text_boundary[0].data.fill_(5.0)
        harness.text_boundary[1].data.fill_(5.0)
        harness.spectral_image_anchors[0].data[0, 0] = 1.0
        harness.spectral_image_anchors[1].data[0, 1] = 1.0
        harness.text_anchors[0].data[0, 0] = 1.0
        harness.text_anchors[1].data[0, 0] = 1.0
        current_spec = torch.zeros(harness.spectral_image_dim)
        current_spec[0] = 1.0
        current_text = torch.zeros(768)
        current_text[0] = 1.0

        scores = harness._score_tasks([0, 1], current_spec, current_text, torch.device("cpu"))

        self.assertGreater(scores[0].item(), scores[1].item())

    def test_role_eval_stage_uses_spectral_image_prototype(self):
        harness = _SpectralHarness()
        harness.active_role_count.data[0] = 2.0
        harness.role_task_count.data[:2] = 1.0
        harness.relation_routing_config["routing_role_member_weight"] = 0.0
        harness.relation_routing_config["routing_role_prior_weight"] = 0.0
        harness.relation_routing_config["routing_role_size_penalty"] = 0.0
        harness.role_spectral_prototypes[0].data[0, 0] = 1.0
        harness.role_spectral_prototypes[1].data[0, 1] = 1.0
        harness.role_text_prototypes[0].data[0, 0] = 1.0
        harness.role_text_prototypes[1].data[0, 0] = 1.0
        image_anchor = torch.zeros(harness.spectral_image_dim)
        image_anchor[0] = 1.0
        text_anchor = torch.zeros(768)
        text_anchor[0] = 1.0

        role_scores = harness._score_roles(
            active_experts=2,
            expert_logits=torch.zeros(2),
            image_anchor=image_anchor,
            text_anchor=text_anchor,
            device=torch.device("cpu"),
        )

        self.assertGreater(role_scores[0].item(), role_scores[1].item())

    def test_early_route_weights_are_smoothed_by_uniform_mix(self):
        harness = _SpectralHarness()
        harness.task_role_membership.data.zero_()
        harness.task_role_membership.data[0, 0] = 1.0
        harness.task_role_membership.data[1, 1] = 1.0
        harness.task_role_membership.data[2, 1] = 1.0
        task_scores = torch.tensor([3.0, 1.0, 0.5], dtype=torch.float32)
        role_weights = torch.tensor([0.999, 0.001], dtype=torch.float32)

        weights = harness._build_early_route_weights(3, task_scores, role_weights)

        self.assertGreater(weights[1].item(), 0.0)
        self.assertGreater(weights[2].item(), 0.0)
        self.assertLess(weights[0].item(), 0.999)

    def test_middle_route_weights_are_smoothed_by_uniform_mix(self):
        harness = _SpectralHarness()
        harness.task_role_membership.data.zero_()
        harness.task_role_membership.data[0, 0] = 1.0
        harness.task_role_membership.data[1, 1] = 1.0
        harness.task_role_membership.data[2, 1] = 1.0
        task_scores = torch.tensor([3.0, 1.5, 1.0], dtype=torch.float32)
        role_weights = torch.tensor([0.999, 0.001], dtype=torch.float32)

        weights = harness._build_middle_route_weights(3, task_scores, role_weights)

        self.assertGreater(weights[1].item(), 0.0)
        self.assertGreater(weights[2].item(), 0.0)
        self.assertLess(weights[0].item(), 0.999)


if __name__ == "__main__":
    unittest.main()
