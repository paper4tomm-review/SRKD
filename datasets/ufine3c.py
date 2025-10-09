import os.path as op
from typing import List, Dict
from utils.iotools import read_json
from .bases import BaseDataset


class UFine3C(BaseDataset):
    """
    UFine3C Evaluation Dataset

    Reference:
    A comprehensive evaluation set with cross-domain, cross-granularity and cross-style characteristics

    Dataset statistics:
    ### identities: 2,250
    ### images: 7,446
    ### text queries: 37,939
    ### split: test-only

    Annotation format:
    {
        "id": int,           # person identity
        "source": str,       # data source
        "file_path": str,    # relative image path
        "captions": List[str] # multiple text descriptions
    }
    """
    dataset_dir = "UFine3C/UFine3C/"

    def __init__(self, datasets_root: str, verbose: bool = True):
        super(UFine3C, self).__init__()

        # Configure paths
        self.dataset_dir = op.join(datasets_root, self.dataset_dir)
        self.anno_path = op.join(self.dataset_dir, "UFine3C_Annotations.json")
        self.img_dir = op.join(self.dataset_dir, "images")

        # Check data integrity
        self._check_before_run()

        # Load all annotations as test set
        self.test_annos = read_json(self.anno_path)

        # Process annotations for evaluation
        self.test, self.test_id_container = self._process_anno()

        if verbose:
            self.logger.info("=> UFine3C evaluation data loaded")
            self.show_dataset_info()

    def _process_anno(self):
        """Process annotations for evaluation"""
        pid_container = set()
        dataset = {
            "image_pids": [],
            "img_paths": [],
            "caption_pids": [],
            "captions": []
        }

        for anno in self.test_annos:
            pid = int(anno["id"])
            pid_container.add(pid)

            # Build absolute image path
            img_rel_path = anno["file_path"]
            img_abs_path = op.join(self.img_dir, op.basename(img_rel_path))

            # Process all captions
            dataset["image_pids"].extend([pid] * len(anno["captions"]))
            dataset["img_paths"].extend([img_abs_path] * len(anno["captions"]))
            dataset["caption_pids"].extend([pid] * len(anno["captions"]))
            dataset["captions"].extend(anno["captions"])

        return dataset, pid_container

    def _check_before_run(self):
        """Verify data integrity"""
        required_paths = [
            self.dataset_dir,
            self.img_dir,
            self.anno_path
        ]

        for path in required_paths:
            if not op.exists(path):
                raise RuntimeError(f"Required path '{path}' not found")

    def show_dataset_info(self):
        """Log dataset statistics"""
        num_ids = len(self.test_id_container)
        num_images = len(self.test_annos)
        num_captions = sum(len(a["captions"]) for a in self.test_annos)

        self.logger.info(f"Evaluation Set Statistics:")
        self.logger.info(f"  {'IDs':<6} | {'Images':<6} | {'Captions':<8}")
        self.logger.info("-" * 30)
        self.logger.info(f"  {num_ids:<6} | {num_images:<6} | {num_captions:<8}")