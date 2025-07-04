import os
from huggingface_hub import login, HfApi, create_repo, upload_folder


def main():
    print("\n=== Hugging Face Dataset Uploader ===\n")
    # Login to Hugging Face
    print("You must be logged in to Hugging Face to upload a dataset.")
    login()

    # Prompt for dataset folder
    while True:
        dataset_dir = input("Enter the path to your dataset folder: ").strip()
        if os.path.isdir(dataset_dir):
            break
        print("Invalid folder. Please enter a valid dataset directory.")

    # Prompt for repo name
    repo_name = input("Enter Hugging Face dataset repo name (e.g., 'username/dataset-name'): ").strip()
    if '/' not in repo_name:
        print("Repo name must be in the format 'username/dataset-name'. Exiting.")
        return

    # Create repo if it doesn't exist
    api = HfApi()
    try:
        create_repo(repo_id=repo_name, repo_type="dataset", exist_ok=True)
    except Exception as e:
        print(f"Failed to create repo: {e}")
        return

    # Upload folder
    print(f"Uploading {dataset_dir} to https://huggingface.co/datasets/{repo_name} ...")
    try:
        upload_folder(
            repo_id=repo_name,
            folder_path=dataset_dir,
            repo_type="dataset",
            commit_message="Add dataset via uploader script"
        )
        print(f"Upload successful! View your dataset at: https://huggingface.co/datasets/{repo_name}")
    except Exception as e:
        print(f"Upload failed: {e}")

if __name__ == "__main__":
    main() 