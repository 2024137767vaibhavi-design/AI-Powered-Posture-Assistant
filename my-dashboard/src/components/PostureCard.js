import "./PostureCard.css";

function PostureCard({ status }) {
  const isCorrect = status === "correct";

  return (
    <div className={`card ${isCorrect ? "correct" : "wrong"}`}>
      <h1>{isCorrect ? "✅ CORRECT POSTURE" : "❌ WRONG POSTURE"}</h1>

      {!isCorrect && (
        <>
          <img
            src="https://i.imgur.com/9Xnq0vP.png"
            alt="Correct posture"
            className="posture-img"
          />
          <p className="tip">
            Keep your back straight, knees aligned, and core tight.
            <br />
            Recommended reps: 12–15
          </p>
        </>
      )}
    </div>
  );
}

export default PostureCard;
